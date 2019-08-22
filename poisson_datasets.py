import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import h5py
import os
from copy import deepcopy

class PoissonSquareRoomOutDataset(Dataset):
    '''
    Dataset that returns sampling of the temperature at different points
    for the squared room Poisson experiments of 'Graph Element Networks' paper.

    Receives path to an HDF5 file with keys of the form i-IN, i-OUT.

    WARNING: to facilitate node position optimization experiments,
             each element is actually a batch of datasets
    '''
    def __init__(self, file_path):
        self.file_path = file_path
        f = h5py.File(file_path, 'r')
        self.NAMES = sorted([int(key[:-3]) for key in f.keys() if 'IN' in key])
        self.DATA = [[f[str(key)+'-IN'][:], f[str(key)+'-OUT'][:]]
                for key in self.NAMES]
        self.NAMES = list(map(str, self.NAMES))
        print(self.NAMES)
        self.size = len(self.DATA)
        assert len(self.NAMES) == self.size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if type(idx) is not int: idx = idx.item()
        return (torch.FloatTensor(self.DATA[idx][0]),
                torch.FloatTensor(self.DATA[idx][1]))

class PoissonSquareRoomInpDataset(Dataset):
    '''
    Dataset that returns sampling of heaters and exterior temperatures
    for the squared room Poisson experiments of 'Graph Element Networks' paper.

    Receives path to a directory of JSON files with room descriptions.
    Dataset i should be in desc_i.json

    WARNING: to facilitate node position optimization experiments,
             each element is actually a batch of datasets
    '''
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.size = 0
        for f in os.listdir(self.dir_path):
            if os.path.isfile(os.path.join(self.dir_path,f)) and 'json' in f:
                self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        '''
        Gets room info from dir_path/desc_[idx].json and transforms it
        '''
        if type(idx) is not int: idx = idx.item()
        with open(os.path.join(self.dir_path, 'desc_' + str(idx)+'.json'), 'r')\
                as infile:
            context = json.load(infile)
        inputs_per_room = 64
        all_inputs = []
        for room in context:
            inputs = []
            #Encode heaters
            for box in room['unnorm_boxes']:
                [lx,ly, Lx, Ly, heat] = box
                inputs.append([lx,ly,0,heat,1])
                inputs.append([lx,Ly,0,heat,1])
                inputs.append([Lx,ly,0,heat,1])
                inputs.append([Lx,Ly,0,heat,1])

                inputs.append([lx,(ly+Ly)/2.,0,heat,1])
                inputs.append([Lx,(ly+Ly)/2.,0,heat,1])
                inputs.append([(lx+Lx)/2.,ly,0,heat,1])
                inputs.append([(lx+Lx)/2.,Ly,0,heat,1])

                inputs.append([(lx+Lx)/2.,(ly+Ly)/2.,0,heat,1])

            #Encode border
            ext_temp = room['exterior_temp']
            num_border = inputs_per_room - len(inputs)
            delta_l = 4./num_border
            for i in range(num_border):
                l = delta_l*i
                l_mod = l
                while l_mod >=1: l_mod -=1
                if l < 1: #top
                  inputs.append([l_mod, 0, ext_temp, 0, 0])
                elif l < 2:
                  inputs.append([1, l_mod, ext_temp, 0, 0])
                elif l < 3:
                  inputs.append([1-l_mod, 1, ext_temp, 0, 0])
                elif l < 4:
                  inputs.append([0, 1-l_mod, ext_temp, 0, 0])
                else: assert False
            all_inputs.append(torch.FloatTensor(np.vstack(inputs)))
        all_inputs = torch.stack(all_inputs)
        return (all_inputs[:,:,:2].contiguous(),
                all_inputs[:,:,2:].contiguous())
