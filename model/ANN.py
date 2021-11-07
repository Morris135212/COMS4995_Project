import torch.nn as nn
import torch.nn.functional as F
import torch


class Model(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.5):
        super(Model, self).__init__()
        self.cls = output_size
        self.fc1 = nn.Linear(input_size, 8192)
        self.fc2 = nn.Linear(8192, 2048)
        self.fc3 = nn.Linear(2048, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, output_size)
        self.drop = dropout

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = F.dropout(X, self.drop)
        X = F.relu(self.fc4(X))
        if self.cls == 1:
            X = torch.sigmoid(self.fc5(X))
        else:
            X = F.softmax(self.fc5(X))
        return X.to(torch.float32)

