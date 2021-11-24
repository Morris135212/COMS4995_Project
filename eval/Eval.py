import numpy as np
import scipy
import torch
from torch.autograd import Variable
from tqdm import tqdm


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


def multi_accuracy_tensor(preds, y):
    _, preds = torch.max(preds, 1)
    correct = (preds == y).sum().item()
    return correct/len(y)


class Evaluator:
    def __init__(self, val_loader, model, device, cls, criterion=torch.nn.BCELoss()):
        self.val = val_loader
        self.device = device
        self.model = model
        self.model.eval()
        self.cls = cls
        self.criterion = criterion
        self.pred = []
        self.labels = []

    def eval(self):
        total_loss, total_acc = 0., 0.
        length = 0
        with torch.no_grad():
            for i, data in enumerate(self.val, 0):
                x, label = data
                x = x.float().to(self.device)
                output = self.model(x)
                if self.cls == 1:
                    try:
                        label = label.float().to(self.device)
                        output = output.squeeze()
                        # print(output.size(), label.size())
                        total_loss += self.criterion(output, label).item()*len(label)
                        total_acc += binary_accuracy_tensor(output, label).item()*len(label)
                        length += len(label)
                        self.labels += list(label.cpu().numpy())
                        self.pred += list(output.cpu().numpy())
                    except Exception as e:
                        print(f"exception: {e}")
                        continue
                else:
                    label = label.to(self.device)
                    total_loss += self.criterion(output, label).item()*len(label)
                    total_acc += multi_accuracy_tensor(output, label)*len(label)
                    length += len(label)
                    length += len(label)
                    self.labels += list(label.cpu().numpy())
                    self.pred += list(output.cpu().numpy())
                del x, label
        # print(self.pred)
        return {"acc": total_acc/length, "loss": total_loss/length}