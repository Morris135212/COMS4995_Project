import numpy as np
import scipy
import torch


def binary_accuracy_tensor(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def binary_accuracy_numpy(preds, y):
    rounded_preds = np.round(preds)
    correct = (rounded_preds == y).astype("float32")
    acc = correct.sum() / len(correct)
    return acc


class Evaluator:
    def __init__(self, val, model, device, cls, criterion):
        self.val_x, self.val_y = val[0], val[1]
        self.device = device
        self.model = model
        self.model.eval()
        self.cls = cls
        self.criterion = criterion
        self.pred = []
        self.labels = self.val_y

    def eval(self):
        x = self.val_x.todense().A if isinstance(self.val_x, scipy.sparse.csr.csr_matrix) else self.val_x
        with torch.no_grad():
            x = torch.from_numpy(x).to(self.device).float()
            y = torch.from_numpy(self.val_y).to(self.device).float()
            preds = self.model(x)
            if self.cls == 1:
                preds = preds.squeeze()
                loss = self.criterion(preds, y).item()
                acc = binary_accuracy_tensor(preds, y)
                self.pred = preds.cpu().numpy()
        del x
        del y
        return {"acc": acc, "loss": loss}