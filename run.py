from dataset.Encoder import DateHourEncoder, DateMonthEncoder, DateWeekEncoder, DateYearEncoder
from dataset.Preprocess import Preprocess, MissingHandler
import numpy as np
from dataset.Sampling import Sampling, SampleMechanism
from eval.Eval import Evaluator
from model.ANN import Model, weights_init, EntityEmbeddingNN
from train.Trainer import Trainer
from utils.read_file import read_from_csv
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    file = "data/transaction.csv"
    df = read_from_csv(file)
    df = df.sample(n=200000, random_state=23)
    df = df.drop(
        ["accountNumber", "accountOpenDate", "dateOfLastAddressChange", "currentExpDate", "transactionDateTime"],
        axis=1)
    handler = MissingHandler(df, impute=False, target="isFraud")
    X, y = handler.get_features(), handler.get_labels()

    # Threeway hold out
    dev_x, test_x, dev_y, test_y = train_test_split(X, y, test_size=0.2, random_state=23, stratify=y)
    train_x, val_x, train_y, val_y = train_test_split(dev_x, dev_y, test_size=0.25, random_state=23, stratify=dev_y)

    p = Preprocess(dev_x, dev_y, handler)
    p.__fit__()

    train_x, train_y = p.preprocessor.transform(train_x), p.tar_handler.transform(train_y)
    val_x, val_y = p.preprocessor.transform(val_x), p.tar_handler.transform(val_y)
    test_x, test_y = p.preprocessor.transform(test_x), p.tar_handler.transform(test_y)

    cate_f = p.col_cate[0]
    num_f = p.col_num[0]
    # print(cate_f, num_f)
    n_uniques = p.preprocessor.transformers_[0][1]['onehot'].categories_
    n_uniques = [len(n) for n in n_uniques]
    n_numeric = len(num_f)
    entityembedding = EntityEmbeddingNN(np.array(n_uniques), n_numeric)
    # print(n_uniques)


"""
if __name__ == "__main__":
    file = "data/transaction.csv"
    df = read_from_csv(file)
    df = df.drop(["accountNumber"], axis=1)
    df = df.sample(n=100000, random_state=23)
    handler = MissingHandler(df, impute=False, target="isFraud")
    X, y = handler.get_features(), handler.get_labels()
    # Threeway hold out
    dev_x, test_x, dev_y, test_y = train_test_split(X, y, test_size=0.2, stratify=y, random_state=23)
    train_x, val_x, train_y, val_y = train_test_split(dev_x, dev_y, test_size=0.25, stratify=dev_y, random_state=23)

    p = Preprocess(X, y, handler)
    p.__fit__()

    train_x, train_y = p.preprocessor.transform(train_x), p.tar_handler.transform(train_y)
    val_x, val_y = p.preprocessor.transform(val_x), p.tar_handler.transform(val_y)
    # samples = Sampling(train_x, train_y, mechanism=SampleMechanism.SMOTE)
    # train_x, train_y = samples.get_features(), samples.get_labels()
    from torch.utils.tensorboard import SummaryWriter
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/fraud')

    trainer = Trainer((train_x, train_y),
                      (val_x, val_y),
                      train_x.shape[1],
                      focal=True,
                      cls=2,
                      writer=writer,
                      lr=1e-3,
                      interval=50)
    trainer.train()
    # val_dataset = CustomDataset(val_x, val_y)
    # val_loader = DataLoader(val_dataset, batch_size=32)
    # model = Model(input_size=train_x.shape[1], output_size=1)
    # e = Evaluator(val_loader, model, device=torch.device("cpu"), cls=1)
    # e.eval()
"""
"""
if __name__=="__main__":
    model = Model(input_size=7000, output_size=2)
    model.apply(weights_init)
    print(model)
"""