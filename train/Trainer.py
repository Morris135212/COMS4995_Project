import torch.nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.Dataset import CustomDataset
from eval.Eval import Evaluator, binary_accuracy_tensor
from model.ANN import Model
from torch.utils.tensorboard import SummaryWriter
from utils.torchtools import EarlyStopping


class Trainer:
    def __init__(self,
                 train: tuple,
                 val: tuple,
                 input_size,
                 cls,
                 writer=SummaryWriter('runs/fashion_mnist_experiment_1'),
                 optimizer="sgd",
                 epochs=10,
                 batch_size=64,
                 lr=1e-5,
                 momentum=1e-5,
                 interval=10,
                 patience=10,
                 path="data/checkpoints/checkpoint.pt"):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        train_dataset = CustomDataset(train[0], train[1])
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.batch_size = batch_size
        self.valset = val
        self.trainset = train
        self.epochs = epochs
        if cls == 1:
            self.criterion = torch.nn.BCELoss()  # Binary cross entropy
        else:
            self.criterion = torch.nn.CrossEntropyLoss()  # Cross Entropy Loss

        self.model = Model(input_size=input_size, output_size=cls)
        self.cls = cls
        if optimizer == "adam":
            self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer == "sgd":
            self.optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.writer = writer
        self.interval = interval
        self.early_stopping = EarlyStopping(patience=patience, verbose=True, path=path)

    def train(self):
        print("Start Training")
        self.model.to(self.device)
        for epoch in range(self.epochs):
            print(f"At epoch: {epoch}")
            total_loss, epoch_loss, epoch_acc = 0., 0., 0.
            length = 0
            for i, data in enumerate(tqdm(self.train_loader), 0):
                self.model.train()

                x, label = data
                x = x.float()
                label = label.float()
                x = Variable(x).to(self.device)
                y = Variable(label).to(self.device)
                output = self.model(x)

                if self.cls == 1:
                    output = output.squeeze()
                    loss = self.criterion(output, y)
                    epoch_loss += loss.item() * len(label)
                    epoch_acc += binary_accuracy_tensor(output, y.squeeze()).item() * len(label)
                    length += len(label)
                else:
                    loss = self.criterion(output, y)
                    """
                    TODO multi-classification
                    """
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if i % self.interval == self.interval - 1:
                    evaluator = Evaluator(self.valset,
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
                del x
                del y
            if self.early_stopping.early_stop:
                break
