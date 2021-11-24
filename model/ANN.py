import torch.nn as nn
import torch.nn.functional as F
import torch


class Model(nn.Module):
    def __init__(self, input_size, output_size, n_hidden1=4096, n_hidden2=1024):
        super(Model, self).__init__()
        self.cls = output_size
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, n_hidden1),
            nn.BatchNorm1d(n_hidden1),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden1, n_hidden2),
            nn.BatchNorm1d(n_hidden2),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden2, output_size)
        )

    def forward(self, X):
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        if self.cls == 1:
            X = torch.sigmoid(X)
        return X

