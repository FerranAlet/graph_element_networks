import numpy as np
import torch
from torch import nn

class NeuralProcesses(nn.Module):
    def __init__(self, encoders, decoders):
        super(NeuralProcesses, self).__init__()
        self.encoders = encoders
        self.decoders = decoders

    def forward(self, Inp, Q):
        '''
        Inp: list of input points (X, y_i) of function i
        Q:   list of queries X for function j
        '''
        aux = []
        #(BS, #inp, feat)
        for (inp, enc) in zip(Inp, self.encoders):
            res = (enc(torch.cat((inp[0], inp[1]), dim=-1)))
            aux.append(res)
        aux = torch.cat(aux, dim=1)
        inp_summ = torch.mean(aux, dim=1, keepdim=True) #[BS, 1, feat]
        res = []
        for (q, dec) in zip(Q, self.decoders):
            dec_inp = torch.cat((inp_summ.repeat(1, q.shape[1], 1), q), dim=2)
            res.append(dec(dec_inp))
        return res
