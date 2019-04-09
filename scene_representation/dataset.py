import random
import math as m

import collections, os, io
from PIL import Image
import torch
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import Dataset
import numpy as np

Context = collections.namedtuple('Context', ['frames', 'cameras'])
Scene = collections.namedtuple('Scene', ['frames', 'cameras'])

def transform_poses(v):
    # Originally, v comes as 3D position + [yaw, pitch]
    w, z = torch.split(v, 3, dim=-1)
    y, p = torch.split(z, 1, dim=-1)
    view_vector = [w, torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)]
    v_hat = torch.cat(view_vector, dim=-1)
    return v_hat

# Own dataset class to be used by torch DataLoader
class Dataset_Custom(Dataset):
    def __init__(self, root_dir, transform=None, force_size=None, allow_multiple_passes=False):
        self.root_dir = root_dir
        self.transform = transform
        self.size = force_size if force_size != None else len(os.listdir(self.root_dir))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        scene_path = os.path.join(self.root_dir, "{}.pt".format(idx))
        data = torch.load(scene_path)

        byte_to_tensor = lambda x: ToTensor()(Resize(64)((Image.open(io.BytesIO(x)))))
        images = torch.stack([byte_to_tensor(frame) for frame in data.frames])
        poses = torch.from_numpy(data.cameras)
        poses = poses.view(-1, 5)
        poses = transform_poses(poses)
        return images, poses


# Grab one batch of supermaze views and queries, 
# Form candidates bucket from all views and queries and get ground truth indices
def sample_batch(x_data=None, v_data=None, expected_bs=None, scenes_per_dim=1, \
    shift=(0.0,0.0), seed=None):

    if seed: random.seed(seed)

    # Convert batch size asked in terms of num of supermazes 
    # to number of total single mazes to be loaded
    num_images_bs = expected_bs * scenes_per_dim**2
    x_data, v_data = x_data[:num_images_bs], v_data[:num_images_bs]
    x_data, v_data = organize_supermaze_data(x_data, v_data, expected_bs, scenes_per_dim, shift)
    scenes_per_supermaze = scenes_per_dim**2

    # N=Total available views per single maze, V=num views per single maze for embedding, Q=num queries per single maze
    N, V, Q = 300, 8, 1
    context_idx = torch.LongTensor([[random.sample(range(N*i, N*(i+1)), V)] for i in range(scenes_per_supermaze)]).reshape(-1)
    query_idx = torch.LongTensor([[random.sample(range(N*i, N*(i+1)), Q)] for i in range(scenes_per_supermaze)]).reshape(-1)
    x, v = x_data[:, context_idx], v_data[:, context_idx]
    x_q, v_q = x_data[:, query_idx], v_data[:, query_idx]

    # Combine all query frames (at the head) and view frames (tail) for the full candidates bucket
    candidates_bucket = torch.cat([x_q.view(-1, 3, 64, 64), x.view(-1, 3, 64, 64)])
    answer_indices = [i for i in range(x_q.shape[0]*x_q.shape[1])]
    return x, v, x_q, v_q, candidates_bucket, answer_indices


def organize_supermaze_data(x_data, v_data, expected_bs, scenes_per_dim, shift):
    bs_received, views = x_data.shape[:2]
    scenes_per_supermaze = int(bs_received / expected_bs)

    # Data into supermazes with the batch size expected by program
    x_data = x_data.view(expected_bs, views*scenes_per_supermaze, 3, 64, 64)
    v_data = v_data.view(expected_bs, views*scenes_per_supermaze, 7)

    # Correction to poses to spread single mazes across grid, for supermazes
    x_corr = np.repeat(torch.arange(0, scenes_per_dim, 1).unsqueeze(0).cpu(), repeats=scenes_per_dim*views).to(v_data.device)
    y_corr = np.repeat(np.repeat(torch.arange(0, scenes_per_dim, 1).unsqueeze(0).cpu(), repeats=views).unsqueeze(0), repeats=scenes_per_dim, axis=0).view(scenes_per_supermaze*views).to(v_data.device)
    xy_corr = torch.cat((x_corr.unsqueeze(1), y_corr.unsqueeze(1)), dim=1)
    xy_corr_per_batch = torch.cat((xy_corr.float(), v_data.new_zeros(scenes_per_supermaze*views, 5)), dim=1)
    xy_corr_all = xy_corr_per_batch.unsqueeze(0).repeat(expected_bs, 1, 1) * 2 # Factor of 2 as images are width 2. height 2.

    # Shift supermaze within a grid of 10x10 (even during training where seen supermazes are up to 2x2)
    shift_tensor = v_data.new_zeros(xy_corr_all.shape[1:])
    shift_tensor[:,0]=shift[0]
    shift_tensor[:,1]=shift[1]
    shift_tensor = shift_tensor.unsqueeze(0).repeat(expected_bs, 1,1)
    v_data += (xy_corr_all + shift_tensor)

    return x_data, v_data
