import torch
from torch.autograd import Variable


def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


class Evaluator:
    def __init__(self, val_loader, model, device, cls, criterion):
        self.val_loader = val_loader
        self.device = device
        self.model = model
        self.model.eval()
        self.cls = cls
        self.criterion = criterion

    def eval(self):
        epoch_loss, epoch_acc = 0., 0.
        length = 0
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                X, y = data
                X = Variable(X.to(torch.float32)).to(self.device)
                y = Variable(y.to(torch.float32)).to(self.device)
                output = self.model(X)
                if self.cls == 1:
                    output = output.squeeze()
                    pred = torch.round(output)
                    epoch_acc += binary_accuracy(pred, y).item() * len(y)
                    epoch_loss += self.criterion(output, y).item() * len(y)
                    length += len(y)

        return {"acc": epoch_acc/length, "loss": epoch_loss/length}
