from itertools import chain

import torch.nn as nn
import torch
import numpy as np


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
        nn.init.kaiming_normal_(m.weight)


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
        return X


class EntityEmbeddingNN(nn.Module):
    def __init__(
            self,
            n_uniques: np.ndarray,
            n_numeric: int,
            A=10, B=5,
            dropout1=0.5,
            dropout2=0.5,
            n_class=2
    ):
        super(EntityEmbeddingNN, self).__init__()
        self.epoch = 0
        self.n_class = n_class
        self.dropout2 = dropout2
        self.dropout1 = dropout1
        self.n_uniques = n_uniques
        self.A = A
        self.B = B
        exp_ = np.exp(-n_uniques * 0.05)
        self.embed_dims = np.round(5 * (1 - exp_) + 1).astype("int")
        sum_ = np.log(self.embed_dims).sum()
        self.n_layer1 = min(1024,
                            int(A * (n_uniques.size ** 0.5) * sum_ + 1)+n_numeric*2)
        self.n_layer2 = int(self.n_layer1 / B) + 2
        self.embeddings = nn.ModuleList([
            nn.Embedding(int(n_unique), int(embed_dim))
            for n_unique, embed_dim in zip(self.n_uniques, self.embed_dims)
        ])
        self.layer1 = nn.Sequential(
            nn.Linear(self.embed_dims.sum()+n_numeric, self.n_layer1),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout1)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(self.n_layer1, self.n_layer2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout2)
        )
        self.dense = nn.Sequential(
            self.layer1,
            self.layer2,
        )
        # regression
        if n_class == 1:
            self.output = nn.Linear(self.n_layer2, 1)
        # binary classification
        elif n_class == 2:
            self.output = nn.Sequential(
                nn.Linear(self.n_layer2, 1),
                nn.Sigmoid()
            )
        # multi classification
        elif n_class > 2:
            self.output = nn.Sequential(
                nn.Linear(self.n_layer2, n_class),
                nn.Softmax()
            )
        else:
            raise ValueError(f"Invalid n_class : {n_class}")
        for m in chain(self.dense.modules(), self.output.modules(), self.embeddings.modules()):
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def forward(self, X):
        embeds = [self.embeddings[i](X[:, i].int())
                  for i in range(len(self.n_uniques))]
        embeds.append(X[:, len(self.n_uniques):])
        # embeds = [self.embeddings[i](torch.from_numpy(X[:, i].astype("int64")))
        #           for i in range(X.shape[1])]
        features = self.dense(torch.cat(embeds, dim=1))
        outputs = self.output(features)
        return embeds, features, outputs