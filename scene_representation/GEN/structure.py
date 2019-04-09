import numpy as np
import torch 

class GraphStructure():

  def __init__(self, num_scenes_per_dim=1, shift=(0,0)):
    # Initialize nodes and edges, grid structure and interpolator placeholders
    self.EPS = np.exp(-10)
    self.width_one_scene = 2.0
    self.shift = shift
    self.msg_steps = 9
    self.node_dim = 256
    self.update_sz = 254
    self.msg_sz = 256
    self.node_interpolator = {} 
    self.initialize_nodes(num_scenes_per_dim)
    self.initialize_edges()
    return

  
  # Build the list of all node positions wihin this grid Structure
  def initialize_nodes(self, num_scenes_per_dim=1):
    self.grid = {}
    self.grid['X'] = self.grid['Y'] = num_scenes_per_dim * self.width_one_scene
    self.grid['min_X'] = -1. + self.shift[0]
    self.grid['min_Y'] = -1. + self.shift[1]
    self.grid['n_x'] = self.grid['n_y'] = 4*(num_scenes_per_dim-1) + 5
    self.grid['dx'] = self.grid['dy'] = self.grid['X'] / (self.grid['n_x']-1)
    
    # Populate node positions
    node_positions = []
    for x in range(self.grid['n_x']):
      for y in range(self.grid['n_y']):
        node_positions.append(np.array([self.grid['min_X']+x*self.grid['dx'], self.grid['min_Y']+y*self.grid['dy']]))

    self.node_positions = torch.FloatTensor(np.vstack(node_positions))
    self.num_nodes = self.node_positions.shape[0]
    return

  
  # Build the list of all directional edges for this grid Structure
  def initialize_edges(self, eps=0.01):
    smallest_dist = 1e9
    for i in range(self.node_positions.shape[0]):
      for j in range(i+1, self.node_positions.shape[0]):
        smallest_dist = min(smallest_dist, np.linalg.norm(
          self.node_positions[i]-self.node_positions[j]))
    
    edges = []
    for i in range(self.node_positions.shape[0]):
      for j in range(i+1, self.node_positions.shape[0]):
        if (np.linalg.norm(self.node_positions[i]-
          self.node_positions[j]) < (1+eps)*smallest_dist):
          edges.append([i,j])
          edges.append([j,i])

    self.edge_sources = [_[0] for _ in edges]
    self.edge_sinks = [_[1] for _ in edges]
    return


  # Given poses, outputs a score tensor indicating how much weight is assigned to each node, per pose
  def get_interpolation_coordinates(self, poses):
    shape = poses.shape

    # Need nx and ny (num units along each dim of grid) to grab the node_positions for each point from input
    # - self.grid['min_X'] or - self.grid['min_Y'] allows yielding positive indices
    nx = ((poses[:,:,0] - self.grid['min_X']) / self.grid['dx']).floor_().long()
    ny = ((poses[:,:,1] - self.grid['min_Y']) / self.grid['dy']).floor_().long()
    node_pos = self.node_positions.to(poses.device)

    # Flatten indices tensors to get 1D tensor for use in index_select
    bottom_left_idx = (nx * self.grid['n_y'] + ny).reshape(shape[0] * shape[1])
    bottom_right_idx = (nx * self.grid['n_y'] + ny + 1).reshape(shape[0] * shape[1])
    top_left_idx = ((nx+1) * self.grid['n_y'] + ny).reshape(shape[0] * shape[1])
    top_right_idx = ((nx+1) * self.grid['n_y'] + ny + 1).reshape(shape[0] * shape[1])
    
    # Grab the meaningful nodes
    bottom_left = torch.index_select(node_pos, dim=0, index=bottom_left_idx).reshape(shape[0], shape[1], 2)
    bottom_right = torch.index_select(node_pos, dim=0, index=bottom_right_idx).reshape(shape[0], shape[1], 2)
    top_left = torch.index_select(node_pos, dim=0, index=top_left_idx).reshape(shape[0], shape[1], 2)
    top_right = torch.index_select(node_pos, dim=0, index=top_right_idx).reshape(shape[0], shape[1], 2)

    # Grab original coordinates of input points
    original_xy = poses[:,:,:2]

    # Each point is in a square, which we normalize to width,height (1,1)
    # The weighting of each point is equal to the area of the rectangle
    # between x and the opposite corner.
    dd = torch.FloatTensor([self.grid['dx'], self.grid['dy']]).to(poses.device)
    bottom_left_score = torch.prod(torch.abs(top_right - original_xy)/dd, dim=2)
    bottom_right_score = torch.prod(torch.abs(top_left - original_xy)/dd, dim=2)
    top_left_score = torch.prod(torch.abs(bottom_right - original_xy)/dd, dim=2)
    top_right_score = torch.prod(torch.abs(bottom_left - original_xy)/dd, dim=2)

    # Initialize a matrix of scores for every node_position in grid (for every frame of every batch)
    scores = torch.zeros(shape[0], shape[1], self.node_positions.shape[0]).to(poses.device)
    
    # Scatter scores along scores tensor for every node_position in grid (for every frame of every batch),
    # based on interpolated node coordinates
    scores.scatter_(dim=2, index=bottom_left_idx.reshape(shape[0],shape[1],1),
      src=torch.unsqueeze(bottom_left_score, dim=2))
    scores.scatter_(dim=2, index=bottom_right_idx.reshape(shape[0],shape[1],1),
      src=torch.unsqueeze(bottom_right_score, dim=2))
    scores.scatter_(dim=2, index=top_left_idx.reshape(shape[0],shape[1],1),
      src=torch.unsqueeze(top_left_score, dim=2))
    scores.scatter_(dim=2, index=top_right_idx.reshape(shape[0],shape[1],1),
      src=torch.unsqueeze(top_right_score, dim=2))
    
    return scores