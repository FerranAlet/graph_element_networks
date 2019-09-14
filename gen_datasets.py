from torch.utils.data import Dataset

class FTDataset(Dataset):
    '''
    Function Transformation Dataset: each element in this (meta)dataset
    is a pair of lists of datasets:
    ([one dataset per input function], [one dataset per output function])
    Each dataset is a pair (X, y_i) and all X leave in the same space.
    '''
    def __init__(self, inp_datasets, inp_datasets_args,
            out_datasets, out_datasets_args,
            idx_list=None, cuda=False):
        '''
        inp_datasets: list of pytorch Dataset classes for inp functions
        inp_datasets_args: list of tuples of arguments for each inp class
        out_datasets: list of pytorch Dataset classes for out functions
        out_datasets_args: list of tuples of arguments for each out class
        '''
        self.InpDatasets = [ D(**a)
                for (D,a) in zip(inp_datasets, inp_datasets_args) ]
        self.n_inp = len(self.InpDatasets)
        self.OutDatasets = [ D(**a)
                for (D,a) in zip(out_datasets, out_datasets_args) ]
        self.n_out = len(self.OutDatasets)
        self.size = len(self.InpDatasets[0])
        for d in self.InpDatasets + self.OutDatasets:
            assert len(d) == self.size, str(len(d)) + ' ' + str(self.size)

        #TODO: check that all X leave in the same space

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        '''
        Returns a pair (Inp,Out) of lists of tensors.
        We call all the appropriate datasets with the same index idx.
        '''
        return ([d[idx] for d in self.InpDatasets],
                [d[idx] for d in self.OutDatasets]), idx

