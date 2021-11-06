import pandas as pd


def read_from_csv(file="https://media.githubusercontent.com/media/Morris135212/"
                       "DataSet/main/transaction/transaction.csv",
                  index_col=0,
                  columns=None):
    if not columns:
        df = pd.read_csv(file, index_col=index_col)
    else:
        df = pd.read_csv(file, index_col=index_col, names=columns)
    print("Load DataFrame!")
    return df
