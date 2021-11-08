import torch.nn as nn
import torch.nn.functional as F
import torch


class Model(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.5):
        super(Model, self).__init__()
        self.cls = output_size
        self.bn_in = nn.BatchNorm1d(input_size)

        self.fc1 = nn.Linear(input_size, 4096)
        self.bn1 = nn.BatchNorm1d(4096)

        self.fc2 = nn.Linear(4096, 1024)
        self.bn2 = nn.BatchNorm1d(1024)

        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, output_size)
        self.drop = dropout

    def forward(self, X):
        X = F.relu(self.fc1(X))

        X = self.bn1(X)
        X = F.relu(self.fc2(X))
        X = F.dropout(X, self.drop)
        X = self.bn2(X)
        X = F.relu(self.fc3(X))
        X = F.dropout(X, self.drop)
        if self.cls == 1:
            X = torch.sigmoid(self.fc4(X))
        else:
            X = F.softmax(self.fc4(X))
        return X

