import torch.nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.Dataset import CustomDataset
from eval.Eval import Evaluator
from model.ANN import Model


class Trainer:
    def __init__(self,
                 train: tuple,
                 val: tuple,
                 input_size,
                 cls,
                 optimizer="sgd",
                 epochs=10,
                 batch_size=64,
                 lr=1e-5,
                 momentum=1e-5):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        train_dataset = CustomDataset(train[0], train[1])
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.batch_size = batch_size
        val_dataset = CustomDataset(val[0], val[1])
        self.val_loader = DataLoader(val_dataset)

        self.epochs = epochs
        if cls==1:
            self.criterion = torch.nn.BCELoss() # Binary cross entropy
        else:
            self.criterion = torch.nn.CrossEntropyLoss() # Cross Entropy Loss

        self.model = Model(input_size=input_size, output_size=cls)
        if optimizer == "adam":
            self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer == "sgd":
            self.optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

    def train(self):
        print("Start Training")
        self.model.to(self.device)
        for epoch in range(self.epochs):
            epoch_loss = []
            for i, data in enumerate(tqdm(self.train_loader), 0):
                x, label = data
                x = x.to(torch.float32)
                label = label.to(torch.float32)
                x = Variable(x).to(self.device)
                y = Variable(label).to(self.device).reshape(self.batch_size, -1)
                self.model.train()

                output = self.model(x)
                loss = self.criterion(output, y)
                epoch_loss.append(loss.item())
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if i % 100 == 99:
                    evaluator = Evaluator(self.val_loader, model=self.model, device=self.device)
                    results = evaluator.eval()
                    acc, auc, recall, precision = results["accuracy"], results["auc"], results["recall"], results["precision"]
                    print(f"At epoch {epoch}, Acc: {acc}, Roc-Auc: {auc}, Recall: {recall}, Precision: {precision}")

