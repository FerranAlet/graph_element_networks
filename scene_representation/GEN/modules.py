import torch.nn as nn
import torch.nn.functional as F
import torch

class NodeModule(nn.Module):
  def __init__(self):
    super(NodeModule, self).__init__()
    self.fc0 = nn.Linear(in_features=512, out_features=512)
    self.fc0bn = nn.BatchNorm1d(512)
    self.fc1 = nn.Linear(in_features=512, out_features=254)
    self.fc1bn = nn.BatchNorm1d(254)

  def forward(self, x):
    x = F.relu(self.fc0bn(self.fc0(x)))
    return self.fc1bn(self.fc1(x))


class EdgeModule(nn.Module):
  def __init__(self):
    super(EdgeModule, self).__init__()
    self.fc0 = nn.Linear(in_features=512, out_features=512)
    self.fc0bn = nn.BatchNorm1d(512)
    self.fc1 = nn.Linear(in_features=512, out_features=256)
    self.fc1bn = nn.BatchNorm1d(256)
  def forward(self, x):
    x = F.relu(self.fc0bn(self.fc0(x)))
    return self.fc1bn(self.fc1(x))


# Pool is re-used in Baseline and GENs
class Pool(nn.Module):
    def __init__(self):
        super(Pool, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=2, stride=2)
        self.conv1bn = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.conv2bn = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv3bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv4bn = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256+7, 256, kernel_size=3, stride=1, padding=1)
        self.conv5bn = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256+7, 128, kernel_size=3, stride=1, padding=1)
        self.conv6bn = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv7bn = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 254, kernel_size=1, stride=1)
        self.conv8bn = nn.BatchNorm2d(254)
        self.pool  = nn.AvgPool2d(16)

    def forward(self, x, v):
        # Resisual connection
        skip_in  = F.relu(self.conv1bn(self.conv1(x)))
        skip_out = F.relu(self.conv2bn(self.conv2(skip_in)))

        r = F.relu(self.conv3bn(self.conv3(skip_in)))
        r = F.relu(self.conv4bn(self.conv4(r))) + skip_out

        # Broadcast
        v = v.view(v.size(0), 7, 1, 1).repeat(1, 1, 16, 16)
        
        # Resisual connection
        # Concatenate
        skip_in = torch.cat((r, v), dim=1)
        skip_out  = F.relu(self.conv5bn(self.conv5(skip_in)))

        r = F.relu(self.conv6bn(self.conv6(skip_in)))
        r = F.relu(self.conv7bn(self.conv7(r))) + skip_out
        r = F.relu(self.conv8bn(self.conv8(r)))
        
        # Pool
        r = self.pool(r)

        return r