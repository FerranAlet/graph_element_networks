import time
from tqdm import tqdm as Tqdm
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from gen_datasets import FTDataset
from poisson_datasets import PoissonSquareRoomInpDataset, \
        PoissonSquareRoomOutDataset
from poisson_square_experiments_utils import *
from neural_processes import NeuralProcesses
from GEN import GEN
from gen_softnn import GENSoftNN
from gen_planargrid import GENPlanarGrid
from utils import Net

torch.manual_seed(0)
cuda = torch.cuda.is_available()
device = torch.device('cuda') if cuda else torch.device('cpu')
model_type = ['GENSoftNN', 'GENPlanarGrid', 'NP'][0]
bs = 8
k = 32
node_train = 16
sqrt_num_nodes_list = [2,3,4,5,6,7]
copies_per_graph = 2
opt_nodes = False
slow_opt_nodes = False #Train node_pos only in part of each "house" data;slower
do_tensorboard = True
# Changed the random initialization because GeneralizedHalton
# doesn't install well on a Docker. We use another simple random initialization.

if not opt_nodes: slow_opt_nodes = False
full_dataset = FTDataset(inp_datasets=[PoissonSquareRoomInpDataset],
        inp_datasets_args = [{'dir_path' : 'data/poisson_inp'}],
        out_datasets = [PoissonSquareRoomOutDataset],
        out_datasets_args = [{'file_path' : 'data/poisson_out.hdf5'}],
        idx_list=None)
train_size = int(0.8*len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset,
        [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1, num_workers=8,
        shuffle=True, drop_last=False)
test_loader = DataLoader(test_dataset,  batch_size=1, num_workers=8,
        shuffle=True, drop_last=False)

encoders = nn.ModuleList([Net(dims=[5,k,k,k])])
decoders = nn.ModuleList([Net(dims=[k+2,k,k,1])])
loss_fn = nn.MSELoss()
if model_type is 'NP':
    model = NeuralProcesses(encoders, decoders)
    mesh_list = mesh_params = [[None] for _ in range(len(full_dataset))]
else:
    assert min(sqrt_num_nodes_list) >= 1
    if model_type == 'GENSoftNN':
        model = GENSoftNN(encoders=encoders, decoders=decoders)
    elif model_type == 'GENPlanarGrid':
        model = GENPlanarGrid(encoders=encoders, decoders=decoders)
    else: raise NotImplementedError
    mesh_list, mesh_params = create_mesh_list(
            num_datasets=len(full_dataset),
            sqrt_num_nodes_list=sqrt_num_nodes_list,
            initialization='random' if opt_nodes else 'uniform',
            copies_per_graph=copies_per_graph, device=device)
    max_mesh_list_elts = max([len(aux) for aux in mesh_list])
if cuda: model.cuda()
opt = torch.optim.Adam(params=model.parameters(), lr=3e-3)
if len(mesh_params):
    mesh_opt = torch.optim.Adam(params=mesh_params, lr=3e-4)
else: mesh_opt = None

if do_tensorboard: writer = SummaryWriter()
else: writer = None

for epoch in Tqdm(range(1000), position=0):
    train_loss = 0. ;  test_loss = 0.
    train_graphs = 0 ; test_graphs = 0
    train_loss_summ = {num**2:[0,0] for num in sqrt_num_nodes_list}
    test_loss_summ = {num**2:[0,0] for num in sqrt_num_nodes_list}
    pos_change_summ = {num**2:[0,0] for num in sqrt_num_nodes_list}
    for g_idx in Tqdm(range(max_mesh_list_elts), position=1):
        for cnt, ((Inp,Out),idx) in enumerate(train_loader):
            if len(mesh_list[idx]) <= g_idx: continue
            G = mesh_list[idx][g_idx]
            if cuda:
                for d in Out:
                    d[0] = d[0].cuda()
                    d[1] = d[1].cuda()
                for d in Inp:
                    d[0] = d[0].cuda()
                    d[1] = d[1].cuda()
            for d in Inp:
                d[0] = d[0].view([-1] + list(d[0].shape[2:]))
                d[1] = d[1].view([-1] + list(d[1].shape[2:]))
            for d in Out:
                d[0] = d[0].view([-1] + list(d[0].shape[2:]))
                d[1] = d[1].view([-1] + list(d[1].shape[2:]))
            Q = [o[0] for o in Out]
            targets = [o[1] for o in Out]
            train_graphs += 1
            if slow_opt_nodes:
                FInp = [[inp[0][:node_train], inp[1][:node_train]]
                        for inp in Inp]
                FQ = [q[:node_train] for q in Q]
                EInp = [[inp[0][node_train:], inp[1][node_train:]]
                        for inp in Inp]
                EQ = [q[node_train:] for q in Q]
                Fpreds = model(FInp, FQ, G=G)
                Epreds = model(EInp, EQ, G=G)
                finetune_losses = [loss_fn(pred,
                    target[:node_train]).unsqueeze(0)
                    for (pred, target) in zip(Fpreds, targets)]
                finetune_loss = torch.sum(torch.cat(finetune_losses))
                exec_losses = [loss_fn(pred,
                    target[node_train:]).unsqueeze(0)
                    for (pred, target) in zip(Epreds, targets)]
                exec_loss = torch.sum(torch.cat(exec_losses))
                mesh_opt.zero_grad()
                finetune_loss.backward()
                mesh_opt.step()
                # project back to square
                graph_update_meshes_after_opt(mesh_list[idx][g_idx],
                        epoch=epoch, writer=writer)
                loss = exec_loss
            else:
                if model_type == 'NP': preds = model(Inp, Q)
                else: preds = model(Inp, Q, G=G)
                losses = [loss_fn(pred, target).unsqueeze(0)
                    for (pred, target) in zip(preds, targets)]
                loss = torch.sum(torch.cat(losses))
            loss.backward()
            train_loss += loss.item()
            train_loss_summ[G.num_nodes][0] += loss.item()
            pos_change_summ[G.num_nodes][0] += (
                    torch.max(torch.abs(G.pos - G.ini_pos)).item())
            train_loss_summ[G.num_nodes][1] += 1
            pos_change_summ[G.num_nodes][1] += 1
            if (cnt % bs == bs-1) or (cnt == len(train_loader)-1):
                opt.step()
                opt.zero_grad()
    if do_tensorboard:
        for num in sqrt_num_nodes_list:
            writer.add_scalar('train/loss-'+str(num**2),
                    train_loss_summ[num**2][0]/train_loss_summ[num**2][1],
                    epoch)
    for cnt, ((Inp,Out),idx) in Tqdm(enumerate(test_loader), position=1):
        if cuda:
            for d in Out:
                d[0] = d[0].cuda()
                d[1] = d[1].cuda()
            for d in Inp:
                d[0] = d[0].cuda()
                d[1] = d[1].cuda()
        for d in Inp:
            d[0] = d[0].view([-1] + list(d[0].shape[2:]))
            d[1] = d[1].view([-1] + list(d[1].shape[2:]))
        for d in Out:
            d[0] = d[0].view([-1] + list(d[0].shape[2:]))
            d[1] = d[1].view([-1] + list(d[1].shape[2:]))
        Q = [o[0] for o in Out]
        targets = [o[1] for o in Out]
        for g_idx, G in enumerate(mesh_list[idx]):
            if model_type == 'NP': preds = model(Inp, Q)
            else: preds = model(Inp, Q, G=G)
            if opt_nodes:
                finetune_losses = [loss_fn(pred[:node_train],
                    target[:node_train]).unsqueeze(0)
                    for (pred, target) in zip(preds, targets)]
                finetune_loss = torch.sum(torch.cat(finetune_losses))
                exec_losses = [loss_fn(pred[node_train:],
                    target[node_train:]).unsqueeze(0)
                    for (pred, target) in zip(preds, targets)]
                exec_loss = torch.sum(torch.cat(exec_losses))
                finetune_loss.backward()
                loss = exec_loss
            else:
                losses = [loss_fn(pred, target).unsqueeze(0)
                    for (pred, target) in zip(preds, targets)]
                loss = torch.sum(torch.cat(losses))
            test_loss += loss.item()
            test_graphs += 1
            test_loss_summ[G.num_nodes][0] += loss.item()
            test_loss_summ[G.num_nodes][1] += 1
    opt.zero_grad() #Don't train Theta on finetune test set when optmizing nodes
    if mesh_opt is not None:
        mesh_opt.step()
        mesh_opt.zero_grad()
        update_meshes_after_opt(mesh_list, epoch=epoch, writer=writer)
    if do_tensorboard:
        for num in sqrt_num_nodes_list:
            writer.add_scalar('test/loss-'+str(num**2),
                    test_loss_summ[num**2][0]/test_loss_summ[num**2][1],epoch)
            if opt_nodes:
                writer.add_scalar('pos_change-'+str(num**2),
                        pos_change_summ[num**2][0]/pos_change_summ[num**2][1],
                        epoch)
    else:
        print(round(train_loss/(max_mesh_list_elts * train_size), 3),
            round(test_loss/(max_mesh_list_elts * test_size), 3))
