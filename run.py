from dataset.Preprocess import Preprocess

if __name__ == "__main__":
    file = "https://media.githubusercontent.com/media/Morris135212/" \
           "DataSet/main/transaction/transaction.csv"
    p = Preprocess(file=file, target="isFraud")
    p.__fit__()
    train_X = p.transform(p.train_x)
    print(train_X)
