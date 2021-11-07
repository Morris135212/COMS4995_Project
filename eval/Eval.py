import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.Dataset import CustomDataset
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score


class Evaluator:
    def __init__(self, val_loader, model, device):
        self.val_loader = DataLoader(val_loader)
        self.device = device
        self.model = model
        self.model.eval()

    def eval(self):
        labels, pred = [], []
        for i, data in enumerate(tqdm(self.val_loader), 0):
            X, y = data
            X = X.to(torch.float32)
            y = y.to(torch.float32)
            X = Variable(X).to(self.device)
            y = Variable(y).to(self.device).reshape(1, -1)
            output = self.model(X)
            pred.append(output[0][0])
            labels.append(y[0])
        Acc = accuracy_score(labels, pred)
        auc = roc_auc_score(labels, pred)
        recall = recall_score(labels, pred)
        precision = precision_score(labels, pred)
        return {"accuracy": Acc, "auc": auc, "recall": recall, "precision": precision}

