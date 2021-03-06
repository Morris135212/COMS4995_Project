import torch.nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.Dataset import CustomDataset
from eval.Eval import Evaluator, binary_accuracy_tensor, multi_accuracy_tensor
from eval.loss import FocalLoss
from model.ANN import Model, weights_init
from torch.utils.tensorboard import SummaryWriter
from utils.torchtools import EarlyStopping


class Trainer:
    def __init__(self,
                 train: tuple,
                 val: tuple,
                 # input_size,
                 cls,
                 model,
                 initialize=True,
                 focal=False,
                 weight=None,
                 writer=SummaryWriter('runs/isFraud'),
                 optimizer="sgd",
                 step_size=500,
                 epochs=10,
                 batch_size=64,
                 lr=1e-5,
                 momentum=0.5,
                 interval=10,
                 patience=10,
                 path="data/checkpoints/checkpoint.pt"):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        train_dataset = CustomDataset(train[0], train[1])
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.batch_size = batch_size
        self.valset = val
        self.val_loader = DataLoader(CustomDataset(val[0], val[1]), batch_size=batch_size)
        self.trainset = train
        self.epochs = epochs
        if cls == 1:
            if focal:
                self.criterion = FocalLoss()
            else:
                self.criterion = torch.nn.BCELoss(size_average=True)  # Binary cross entropy
            self.criterion.to(device=self.device)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight)  # Cross Entropy Loss
            self.criterion.to(device=self.device)

        # self.model = Model(input_size=input_size, output_size=cls)
        self.model = model
        if initialize:
            self.model.apply(weights_init)
        self.model.to(self.device)
        self.cls = cls
        if optimizer == "adam":
            self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer == "sgd":
            self.optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=step_size, gamma=0.5)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, 0.5)
        self.writer = writer
        self.interval = interval
        self.early_stopping = EarlyStopping(patience=patience, verbose=True, path=path)

    def train(self):
        print("Start Training")
        for epoch in range(self.epochs):
            print(f"At epoch: {epoch}")
            epoch_loss, epoch_acc = 0., 0.
            length = 0
            for i, data in enumerate(tqdm(self.train_loader), 0):
                self.model.train()
                x, label = data
                x = x.float()
                x = Variable(x).to(self.device)
                output = self.model(x)
                if self.cls == 1:
                    label = label.float()
                    y = Variable(label).to(self.device)
                    try:
                        output = output.squeeze()
                    except Exception as e:
                        output = output[2].squeeze()
                    loss = self.criterion(output, y)
                    epoch_loss += loss.item() * len(label)
                    epoch_acc += binary_accuracy_tensor(output, y.squeeze()).item() * len(label)
                    length += len(label)
                else:
                    y = Variable(label).to(self.device)
                    if isinstance(output, tuple):
                        output =output[2]
                    loss = self.criterion(output, y)
                    epoch_loss += loss.item() * len(label)
                    epoch_acc += multi_accuracy_tensor(output, y)*len(label)
                    length += len(label)
                    """
                    TODO multi-classification
                    """
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.scheduler.step()
                del x
                if i % self.interval == self.interval - 1:
                    evaluator = Evaluator(self.val_loader,
                                          model=self.model,
                                          device=self.device,
                                          cls=self.cls,
                                          criterion=self.criterion)
                    results = evaluator.eval()
                    eval_acc, eval_loss = results["acc"], results["loss"]
                    print(f"At epoch {epoch}, eval Acc: {eval_acc}, train Acc: {epoch_acc / length},"
                          f"train loss: {epoch_loss / length}, val loss: {eval_loss}")

                    self.writer.add_scalars('loss/', {"eval_loss": eval_loss,
                                                      "train_loss": epoch_loss / length},
                                            epoch * len(self.train_loader) + i)
                    self.writer.add_scalars('accuracy/', {"eval_acc": eval_acc,
                                                          "train_acc": epoch_acc / length},
                                            epoch * len(self.train_loader) + i)
                    self.early_stopping(eval_loss, self.model)
            if self.early_stopping.early_stop:
                break
