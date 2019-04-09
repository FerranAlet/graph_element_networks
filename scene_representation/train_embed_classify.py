import argparse
import os
import random 
import math as m

import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from dataset import *
from embed_classify import Classify

# Given the pre-softmax matrix and the targets:
# returns the accuracy as a percentage
# returns the argmax indices, corresponding to predictions made 
def get_accuracy(res, targets):
    s = torch.nn.Softmax(dim=1)(res)
    _, indices = torch.max(s, dim=1)
    correct = torch.sum(torch.LongTensor(targets)==indices.cpu()).float().item()
    return (correct / len(targets))*100, indices, s

# Logging intervals
heavy_log_interval = 1000
log_interval = 1000
save_interval = 1000

# Vary data set sizes for debugging
force_size_train = 100000
force_size_test = 10000

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GEN v/s Baseline GQN embedders')
    parser.add_argument('--baseline', type=bool, default=False, help='baseline model? (if False, use GEN)')
    parser.add_argument('--train_bs', type=int, default=4, help='size of training batch (default: 4)')
    parser.add_argument('--test_bs', type=int, default=1, help='size of testing batch (default: 1)')
    parser.add_argument('--train_data_dir', type=str, help='location of training data')
    parser.add_argument('--test_data_dir', type=str, help='location of test data')
    parser.add_argument('--root_log_dir', type=str, help='root location of logs')
    parser.add_argument('--log_dir', type=str, help='log directory (default: Classify)', default='Classify')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=128)
    parser.add_argument('--device_ids', type=int, nargs='+', help='list of CUDA devices (default: [0,1])', default=[0,1])
    parser.add_argument('--saved_model', type=str, help='path to model', default=None)
    parser.add_argument('--seed', type=int, default=3, help='seed value (default: 3)')
    args = parser.parse_args()

    assert(args.train_data_dir)
    assert(args.test_data_dir)
    assert(args.root_log_dir)

    log_dir = os.path.join(args.root_log_dir, args.log_dir)
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, 'models')): os.makedirs(os.path.join(log_dir, 'models'))
    if not os.path.exists(os.path.join(log_dir,'runs')): os.makedirs(os.path.join(log_dir,'runs'))
    
    writer = SummaryWriter(log_dir=os.path.join(log_dir,'runs'))

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    min_train_structure_dim = 1
    max_train_structure_dim = 2
    min_test_structure_dim = 1
    max_test_structure_dim = 5
    interval_alter_structure_train = 1
    train_structure_refresh_needed = True

    scenes_per_dim_train = None
    scenes_per_dim_test = None
    shift_train = (0.0, 0.0)
    shift_test = (0.0, 0.0)

    B = args.train_bs
    B_test = args.test_bs
    loader_bs = [None, None]
    loader_bs[0] = B * max_train_structure_dim**2
    loader_bs[1] = B_test * max_test_structure_dim**2 

    # For parallel model, want batch size to exceed number of GPUs
    assert(loader_bs[0] >= len(args.device_ids) and loader_bs[1] >= len(args.device_ids))

    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir
    train_dataset = Dataset_Custom(root_dir=train_data_dir, force_size=force_size_train, allow_multiple_passes=False)
    test_dataset = Dataset_Custom(root_dir=test_data_dir, force_size=force_size_test, allow_multiple_passes=False)

    kwargs = {'num_workers':args.workers, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = DataLoader(train_dataset, batch_size=loader_bs[0], shuffle=True, drop_last=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=loader_bs[1], shuffle=True, drop_last=True, num_workers=1)
    train_iter = iter(train_loader)
    test_iter = iter(test_loader)

    device = f"cuda:{args.device_ids[0]}" if torch.cuda.is_available() else "cpu"
    baseline = args.baseline
    model = Classify(baseline=baseline).to(device)
    if len(args.device_ids)>1: model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
    loss_fn = torch.nn.CrossEntropyLoss()

    restoring_epoch = 0
    if args.saved_model != None: 
        checkpoint = torch.load(args.saved_model)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        restoring_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])

    total_epochs = int(1e6)

    for t in tqdm(range(total_epochs)):
        i = t + restoring_epoch

        try: x_data, v_data = next(train_iter)
        except:
            train_iter = iter(train_loader)
            x_data, v_data = next(train_iter)

        x_data = x_data.to(device)
        v_data = v_data.to(device)

        if t%interval_alter_structure_train==0 or train_structure_refresh_needed:
            # Re-sample the training supermaze size and the shift within the larger 10.0x10.0 square
            scenes_per_dim_train = random.randint(min_train_structure_dim, max_train_structure_dim)
            space_left = 10.0 - (scenes_per_dim_train*2.0)
            shift_train = (random.uniform(0,space_left), random.uniform(0,space_left))
            if len(args.device_ids)>1: model.module.composer.refresh_structure(scenes_per_dim_train, shift_train)
            else: model.composer.refresh_structure(scenes_per_dim_train, shift_train)
            train_structure_refresh_needed = False

        # Always load B * max_train_structure_dim**2 mazes for consistent size of candidates set
        # Adjust to form the right number of supermazes without leaving out any maze
        adjusted_bs = m.floor((B * (max_train_structure_dim**2)) / (scenes_per_dim_train**2))
        x, v, x_q, v_q, candidates_bucket, answer_indices = \
            sample_batch(x_data=x_data, v_data=v_data, expected_bs=adjusted_bs, \
                scenes_per_dim=scenes_per_dim_train, shift=shift_train)

        model.train()
        embedded_queries, embedded_candidates = model(x, v, v_q, candidates_bucket)
        queries_norm = torch.norm(embedded_queries, dim=1, keepdim=True)**2
        candidates_norm = (torch.norm(embedded_candidates, dim=1, keepdim=True)**2).transpose(0,1)
        cross_product = torch.mm(embedded_queries, embedded_candidates.transpose(1,0))
        res = - (queries_norm.expand_as(cross_product) + candidates_norm.expand_as(cross_product) - 2* cross_product)
        if len(args.device_ids)>1: res *= torch.exp(model.module.scalar).expand_as(res).to(res.device)
        else: res *= torch.exp(model.scalar).expand_as(res).to(res.device)

        cross_entropy_loss = loss_fn(res, torch.LongTensor(answer_indices).to(res.device))
        cross_entropy_loss.backward() 
        optimizer.step()
        optimizer.zero_grad()

        accuracy, indices_predicted, _ = get_accuracy(res, answer_indices)
        writer.add_scalar(f'train loss {scenes_per_dim_train}x{scenes_per_dim_train}', cross_entropy_loss.item(), i)
        writer.add_scalar(f'train accuracy {scenes_per_dim_train}x{scenes_per_dim_train}', accuracy, i)
        
        if i % heavy_log_interval == 0:
            writer.add_image(f'train ground truth {scenes_per_dim_train}x{scenes_per_dim_train}', make_grid(x_q.view(-1,3,64,64), 6, pad_value=1), i)
            writer.add_image(f'train predictions {scenes_per_dim_train}x{scenes_per_dim_train}', make_grid(candidates_bucket[indices_predicted], 6, pad_value=1), i)

        with torch.no_grad():
            if i % log_interval == 0:
                try: x_data_test_raw, v_data_test_raw = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    x_data_test_raw, v_data_test_raw = next(test_iter)

                # 1x1 supermaze to 5x5 supermaze during testing
                for scenes_per_dim_test in range(min_test_structure_dim, max_test_structure_dim+1):      
                    
                    # Re-sample the testing supermaze size and the shift within the larger 10.0x10.0 square              
                    space_left = 10.0 - (scenes_per_dim_test*2.0)
                    shift_test = (random.uniform(0,space_left), random.uniform(0,space_left)) 
                    if len(args.device_ids)>1: model.module.composer.refresh_structure(scenes_per_dim_test, shift_test)
                    else: model.composer.refresh_structure(scenes_per_dim_test, shift_test)
                    
                    x_data_test = x_data_test_raw.clone().to(device)
                    v_data_test = v_data_test_raw.clone().to(device)
                    
                    # Always load B * max_train_structure_dim**2 mazes for consistent size of candidates set
                    # Adjust to form the right number of supermazes without leaving out any maze
                    adjusted_bs = m.floor((B_test * (max_test_structure_dim**2)) / (scenes_per_dim_test**2))

                    x_test, v_test, x_q_test, v_q_test, candidates_bucket, answer_indices = \
                        sample_batch(x_data=x_data_test, v_data=v_data_test, expected_bs=adjusted_bs, \
                        scenes_per_dim=scenes_per_dim_test, shift=shift_test)                    

                    embedded_queries, embedded_candidates = model(x_test, v_test, v_q_test, candidates_bucket)
                    queries_norm = torch.norm(embedded_queries, dim=1, keepdim=True)**2
                    candidates_norm = (torch.norm(embedded_candidates, dim=1, keepdim=True)**2).transpose(0,1)
                    cross_product = torch.mm(embedded_queries, embedded_candidates.transpose(1,0))
                    res = - (queries_norm.expand_as(cross_product) + candidates_norm.expand_as(cross_product) - 2* cross_product)
                    if len(args.device_ids)>1: res *= torch.exp(model.module.scalar).expand_as(res).to(res.device)
                    else: res *= torch.exp(model.scalar).expand_as(res).to(res.device)
                    
                    cross_entropy_loss = loss_fn(res, torch.LongTensor(answer_indices).to(res.device))
                    accuracy, indices_predicted,s = get_accuracy(res, answer_indices)
                    
                    writer.add_scalar(f'test loss {scenes_per_dim_test}x{scenes_per_dim_test}', cross_entropy_loss.item(), i)
                    writer.add_scalar(f'test accuracy {scenes_per_dim_test}x{scenes_per_dim_test}', accuracy, i)
                    writer.add_image(f'test ground truth {scenes_per_dim_test}x{scenes_per_dim_test}', make_grid(x_q_test.view(-1,3,64,64), 6, pad_value=1), i)
                    writer.add_image(f'test predictions {scenes_per_dim_test}x{scenes_per_dim_test}', make_grid(candidates_bucket[indices_predicted], 6, pad_value=1), i)

        train_structure_refresh_needed = True

        if i % save_interval == 0:
            torch.save({ 
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, log_dir + "/models/checkpoint-{}.pt".format(i))

    torch.save({
        'epoch': total_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, log_dir + "/models/checkpoint-{}.pt".format(i))
    writer.close()
