import torch
from torch import nn
from torch.nn import functional as F

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
        skip_in = torch.cat((r, v), dim=1)
        skip_out  = F.relu(self.conv5bn(self.conv5(skip_in)))

        r = F.relu(self.conv6bn(self.conv6(skip_in)))
        r = F.relu(self.conv7bn(self.conv7(r))) + skip_out
        r = F.relu(self.conv8bn(self.conv8(r)))
        
        r = self.pool(r)

        return r


class InferenceCore(nn.Module):
    def __init__(self, num_copies=0):
        super(InferenceCore, self).__init__()
        self.downsample_x = nn.Conv2d(3, 3, kernel_size=4, stride=4, padding=0, bias=False)
        self.upsample_v = nn.ConvTranspose2d(7, 7, kernel_size=16, stride=16, padding=0, bias=False)
        self.upsample_r = nn.ConvTranspose2d(256+(7*num_copies), 256, kernel_size=16, stride=16, padding=0, bias=False)
        self.downsample_u = nn.Conv2d(128, 128, kernel_size=4, stride=4, padding=0, bias=False)
        self.core = Conv2dLSTMCell(3+7+256+2*128, 128, kernel_size=5, stride=1, padding=2)
        
    def forward(self, x_q, v_q, r, c_e, h_e, h_g, u):
        x_q = self.downsample_x(x_q)
        v_q = self.upsample_v(v_q.view(-1, 7, 1, 1))

        if r.size(2)!=h_e.size(2):
            r = self.upsample_r(r)

        u = self.downsample_u(u)
        c_e, h_e = self.core(torch.cat((x_q, v_q, r, h_g, u), dim=1), (c_e, h_e))

        return c_e, h_e
    
    
class GenerationCore(nn.Module):
    def __init__(self, num_copies=0):
        super(GenerationCore, self).__init__()
        self.upsample_v = nn.ConvTranspose2d(7, 7, kernel_size=16, stride=16, padding=0, bias=False)
        self.upsample_r = nn.ConvTranspose2d(256+(7*num_copies), 256, kernel_size=16, stride=16, padding=0, bias=False)
        self.core = Conv2dLSTMCell(7+256+3, 128, kernel_size=5, stride=1, padding=2)
        self.upsample_h = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4, padding=0, bias=False)
        
    def forward(self, v_q, r, c_g, h_g, u, z):
        v_q = self.upsample_v(v_q.view(-1, 7, 1, 1))
        if r.size(2)!=h_g.size(2):
            r = self.upsample_r(r)

        c_g, h_g = self.core(torch.cat((v_q, r, z), dim=1), (c_g, h_g))
        u = self.upsample_h(h_g) + u
        
        return c_g, h_g, u


class Conv2dLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2dLSTMCell, self).__init__()

        kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding)
        in_channels += out_channels
        
        self.forget = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.input  = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.output = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.state  = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, input, states):
        (cell, hidden) = states
        input = torch.cat((hidden, input), dim=1)
        
        forget_gate = torch.sigmoid(self.forget(input))
        input_gate  = torch.sigmoid(self.input(input))
        output_gate = torch.sigmoid(self.output(input))
        state_gate  = torch.tanh(self.state(input))

        # Update internal cell state
        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * torch.tanh(cell)

        return cell, hidden