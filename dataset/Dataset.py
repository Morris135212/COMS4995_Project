from torch.utils.data import Dataset

from dataset.Preprocess import Preprocess
from utils.read_file import read_from_csv


class CustomDataset(Dataset):
    """
    Read a Dataset from a custom csv file
    """
    def __init__(self, file, target, preprocess):
        self.df = read_from_csv(file)
        self.p = preprocess
        self.y = self.df[target]
        self.X = self.df.drop([target], axis=1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        pass

