import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, dims):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.dims = dims
        for i in range(len(self.dims)-1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i+1]))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i+1 < len(self.layers):
                x = F.relu(x)
        return x
