import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return torch.from_numpy(f['lr'][str(idx)][:,:]).permute(2,0,1).float(), torch.from_numpy(f['hr'][str(idx)][:,:]).permute(2,0,1).float(), torch.from_numpy(f['lr_up'][str(idx)][:,:]).permute(2,0,1).float()

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return torch.from_numpy(f['lr'][str(idx)][:,:]).permute(2,0,1).float(), torch.from_numpy(f['hr'][str(idx)][:,:]).permute(2,0,1).float(), torch.from_numpy(f['lr_up'][str(idx)][:,:]).permute(2,0,1).float()

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class TrainDatasetDnCNN(Dataset):
    def __init__(self, h5_file):
        super(TrainDatasetDnCNN, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return torch.from_numpy(f['clean'][str(idx)][:,:]).unsqueeze(0).float(), torch.from_numpy(f['noise'][str(idx)][:,:]).unsqueeze(0).float()

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['clean'])


class EvalDatasetDnCNN(Dataset):
    def __init__(self, h5_file):
        super(EvalDatasetDnCNN, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return torch.from_numpy(f['clean'][str(idx)][:,:]).unsqueeze(0).float(), torch.from_numpy(f['noise'][str(idx)][:,:]).unsqueeze(0).float()

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['clean'])
