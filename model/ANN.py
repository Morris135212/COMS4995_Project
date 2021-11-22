import torch.nn as nn
import torch.nn.functional as F
import torch


class Model(nn.Module):
    def __init__(self, input_size, output_size, n_hidden1=4096, n_hidden2=1024, n_hidden3=256):
        super(Model, self).__init__()
        self.cls = output_size
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, n_hidden1),
            nn.BatchNorm1d(n_hidden1),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden1, n_hidden2),
            nn.BatchNorm1d(n_hidden2),
            nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden2, n_hidden3),
            nn.BatchNorm1d(n_hidden3),
            nn.ReLU(True)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(n_hidden3, output_size)
        )

    def forward(self, X):
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)
        if self.cls == 1:
            X = torch.sigmoid(X)
        else:
            X = F.softmax(X)
        return X

