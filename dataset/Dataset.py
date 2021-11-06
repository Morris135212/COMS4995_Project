from torch.utils.data import Dataset

from utils.read_file import read_from_csv


class CustomDataset(Dataset):
    """
    Read a Dataset from a custom csv file
    """
    def __init__(self, file, target):
        assert file.endswith("csv") or file.endswith("txt"), "Not a required data type"
        self.df = read_from_csv(file)
        assert target in self.df, "Target column not in given dataframe"
        self.y = self.df[target]
        self.X = self.df.drop([target], axis=1)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, item):
        pass
