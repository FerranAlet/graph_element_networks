import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from GEN import GEN


class GENPlanarGrid(GEN):
    def __init__(self, **kwargs):
        super(GENPlanarGrid, self).__init__(**kwargs)
        self.grid_info = None
        self._EPS = 1e-7

    def repr_fn(self, node_pos, x_inp, **kwargs):
        return self.compute_coordinates_planar_grid(node_pos, x_inp, **kwargs)

    def set_grid_info(self, grid_info):
        self.grid_info = grid_info

    def compute_coordinates_planar_grid(self, node_pos, x, grid_info=None):
        if grid_info is None: grid_info = self.grid_info
        assert grid_info is not None
        #Take out batch dimension
        bs = 1 if len(x.shape) == 2 else x.shape[0]
        inps_per_elt, features = x.shape[-2], x.shape[-1]
        pos = x.reshape(-1,features)[:,:2]
        #Find the correct square
        nx = torch.nn.functional.relu(
          (pos[:,0]-self._EPS)/grid_info['dx']).floor_().long()
        ny = torch.nn.functional.relu(
          (pos[:,1]-self._EPS)/grid_info['dy']).floor_().long()
        bottom_left_idx = nx * grid_info['n_y'] + ny
        bottom_left = torch.index_select(node_pos, dim=0, index=bottom_left_idx)
        bottom_right_idx = nx * grid_info['n_y'] + ny + 1
        bottom_right = torch.index_select(
                node_pos, dim=0, index=bottom_right_idx)
        top_left_idx = (nx+1) * grid_info['n_y'] + ny
        top_left = torch.index_select(node_pos, dim=0, index=top_left_idx)
        top_right_idx = (nx+1) * grid_info['n_y'] + ny + 1
        top_right = torch.index_select(node_pos, dim=0, index=top_right_idx)
        # Here we use a math trick to compute the weightings
        # each point is in a square, which we'll normalize to width,height (1,1)
        # The weighting of each point is equal to the area of the rectangle
        # between pos and the opposite corner.
        dd = torch.FloatTensor([
            grid_info['dx'], grid_info['dy']]).to(device=top_right.device)
        bottom_left_score = torch.prod(torch.abs(top_right - pos)/dd, dim=1)
        bottom_right_score = torch.prod(torch.abs(top_left - pos)/dd, dim=1)
        top_left_score = torch.prod(torch.abs(bottom_right - pos)/dd, dim=1)
        top_right_score = torch.prod(torch.abs(bottom_left - pos)/dd, dim=1)
        scores = torch.zeros(pos.shape[0], node_pos.shape[0]).to(device=dd.device)
        scores.scatter_(dim=1, index=torch.unsqueeze(bottom_left_idx, dim=1),
        src=torch.unsqueeze(bottom_left_score, dim=1))
        scores.scatter_(dim=1, index=torch.unsqueeze(bottom_right_idx, dim=1),
        src=torch.unsqueeze(bottom_right_score, dim=1))
        scores.scatter_(dim=1, index=torch.unsqueeze(top_left_idx, dim=1),
        src=torch.unsqueeze(top_left_score, dim=1))
        scores.scatter_(dim=1, index=torch.unsqueeze(top_right_idx, dim=1),
        src=torch.unsqueeze(top_right_score, dim=1))
        return scores.reshape((bs, inps_per_elt, -1))

    def forward(self, Inp, Q, G=None, msg_steps=None, repr_fn_args={}):
        if G is not None: self.set_grid_info(G.grid_info)
        return super(GENPlanarGrid, self).__init__(**kwargs)
