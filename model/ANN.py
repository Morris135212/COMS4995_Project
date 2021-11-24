import torch.nn as nn
import torch


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        print("Initialize Bathch")
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        print("Initialize Linear")
        nn.init.kaiming_normal(m.weight)


class Model(nn.Module):
    def __init__(self, input_size, output_size, n_hidden1=4096, n_hidden2=1024):
        super(Model, self).__init__()
        self.cls = output_size
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, n_hidden1),
            nn.BatchNorm1d(n_hidden1),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden1, n_hidden2),
            nn.BatchNorm1d(n_hidden2),
            nn.ReLU()
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
        else:
            X = torch.softmax(X, dim=1)
        return X

