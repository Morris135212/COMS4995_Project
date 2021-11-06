from dataset.Preprocess import Preprocess
from utils.read_file import read_from_csv

if __name__ == "__main__":
    file = "https://media.githubusercontent.com/media/Morris135212/" \
           "DataSet/main/transaction/transaction.csv"
    df = read_from_csv(file)
    p = Preprocess(df, target="isFraud")
    p.__fit__()
    X = p.transform(p.X)
    print(X)
