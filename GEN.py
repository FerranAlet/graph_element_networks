import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GraphConv, GCNConv


class GEN(nn.Module):
    def __init__(self, encoders, decoders, G=None, msg_steps=None):
        super(GEN, self).__init__()
        self.encoders = encoders
        self.decoders = decoders
        self.G = G if G is not None else Data()
        self.msg_steps = msg_steps
        #should match w/ num_features = num_node_features
        self.G.num_feat = self.encoders[0].layers[-1].out_features
        self.G.num_dimensions = (
                self.G.pos.shape[-1] if self.G.pos is not None else
                self.decoders[0].layers[0].in_features - self.G.num_feat)
        for enc in self.encoders:
            assert enc.layers[-1].out_features == self.G.num_feat
        for dec in self.decoders:
            assert (dec.layers[0].in_features ==
                    self.G.num_feat + self.G.num_dimensions)
        self.conv = GCNConv(self.G.num_feat + self.G.num_dimensions,
                self.G.num_feat) #position shouldn't be touched
        self.layer_norm = nn.modules.normalization.LayerNorm(self.G.num_feat)

    def set_node_pos(self, node_pos):
        self.G.pos = node_pos

    def set_msg_steps(self, msg_steps):
        self.msg_steps = msg_steps

    def forward(self, Inp, Q, G=None, repr_fn_args={}):
        '''
        Inp: list of input points (X, y_i) of function i
        Q:   list of queries X for function j
        '''
        if G is None: G = self.G
        else:
            G.num_feat = self.G.num_feat
            G.num_dimensions = self.G.num_dimensions
        assert G.pos is not None
        if hasattr(G, 'msg_steps'): msg_steps = G.msg_steps
        if msg_steps is None:
            if self.msg_steps is not None: msg_steps = self.msg_steps
            else: msg_steps = G.num_nodes*2-1
        # Encode all inputs
        inputs = [] #(BS, #inp, feat)
        for (inp, enc) in zip(Inp, self.encoders):
            res = (enc(torch.cat((inp[0], inp[1]), dim=-1)))
            inputs.append(res)
        inputs = torch.cat(inputs, dim=1)
        x_inp = torch.cat([inp[0] for inp in Inp], dim=1)
        # Initialize GNN node states with representation function
        inp_coord = self.repr_fn(G.pos, x_inp, **repr_fn_args)
        G.x = torch.bmm(torch.transpose(inp_coord, 1, 2), inputs)
        bs, num_nodes, f = G.x.shape
        # Create Batch to feed to GNN
        data_list = [Data(x=x.squeeze(0), pos=G.pos, edge_index=G.edge_index)
                for x in torch.split(G.x,split_size_or_sections=1,dim=0)]
        SG = Batch.from_data_list(data_list)

        # Propagate GNN states with message passing
        for step in range(msg_steps):
            SG.x = self.layer_norm(SG.x + self.conv(
                torch.cat((SG.pos, SG.x), dim=-1), SG.edge_index))
        G.x = SG.x.reshape((SG.num_graphs,-1,f))

        queries = [] #(BS, #out, feat)
        # Decode hidden states to final outputs
        res = []
        for (q, dec) in zip(Q, self.decoders):
            q_coord = self.repr_fn(G.pos, q, **repr_fn_args)
            lat = torch.bmm(q_coord, G.x)
            res.append(dec(torch.cat((lat, q), dim=-1)))
        return res

    def repr_fn(self, **kwargs):
        raise NotImplementedError("the default GEN class doesn't have \
                the repr_fn implemented, a reasonable default is GENSoftNN")
