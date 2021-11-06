from torch.utils.data import Dataset

from dataset.Preprocess import Preprocess
from utils.read_file import read_from_csv


class CustomDataset(Dataset):
    """
    Read a Dataset from a custom csv file
    """
    def __init__(self, file, target):
        self.df = read_from_csv(file)
        self.p = Preprocess(self.df, target=target)
        self.p.__fit__()
        self.X, self.y = p.X, p.y

    def __len__(self):
        return X.shape[0]

    def __getitem__(self, item):
        pass
