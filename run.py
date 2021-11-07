import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset.Dataset import CustomDataset
from dataset.Preprocess import Preprocess, MissingHandler
from model.ANN import Model
from train.Trainer import Trainer
from utils.read_file import read_from_csv
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    file = "data/transaction.csv"
    df = read_from_csv(file)
    df = df.drop(["accountNumber", "customerId"], axis=1)
    handler = MissingHandler(df, target="isFraud")
    X, y = handler.get_features(), handler.get_labels()
    # Threeway hold out
    dev_x, test_x, dev_y, test_y = train_test_split(X, y, test_size=0.2)
    train_x, val_x, train_y, val_y = train_test_split(dev_x, dev_y, test_size=0.25)

    p = Preprocess(dev_x, dev_y, handler)
    p.__fit__()

    train_x, train_y = p.preprocessor.transform(train_x), p.tar_handler.transform(train_y)
    val_x, val_y = p.preprocessor.transform(val_x), p.tar_handler.transform(val_y)
    trainer = Trainer((train_x, train_y), (val_x, val_y), train_x.shape[1], cls=1)
    trainer.train()

"""
if __name__ == "__main__":
    file = "data/transaction.csv"
    df = read_from_csv(file)
    df = df.drop(["accountNumber", "customerId"], axis=1)
    handler = MissingHandler(df, target="isFraud")
    X, y = handler.get_features(), handler.get_labels()
    # Threeway hold out
    dev_x, test_x, dev_y, test_y = train_test_split(X, y, test_size=0.2)
    train_x, val_x, train_y, val_y = train_test_split(dev_x, dev_y, test_size=0.25)

    p = Preprocess(dev_x, dev_y, handler)
    p.__fit__()

    demo_x, demo_y = p.preprocessor.transform(test_x), p.tar_handler.transform(test_y)
    print(demo_y.shape)
    train_dataset = CustomDataset(demo_x, demo_y)
    train_loader = DataLoader(train_dataset, shuffle=True)
    demo = next(iter(train_loader))[1]
    print(demo)
    # demo = demo.to(torch.float32)
    # model = Model(input_size=demo.shape[1], output_size=1)
    # out = model(demo)
    # print(out.shape)
"""