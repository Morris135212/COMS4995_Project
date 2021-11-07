import torch
from torch.utils.data import Dataset
import scipy
import numpy as np


class CustomDataset(Dataset):
    """
    Read a Dataset from a custom csv file
    """
    def __init__(self, X, y):
        self.y = y
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx, :]
        x = x.todense().A.reshape(-1) if isinstance(x, scipy.sparse.csr.csr_matrix) else x.reshape(-1)
        label = self.y[idx]
        label = label.todense().A if isinstance(label, scipy.sparse.csr.csr_matrix) else label
        return x, label

