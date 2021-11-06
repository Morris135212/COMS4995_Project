from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from utils.read_file import read_from_csv


def missing_handler(df, target, thresh=0.25, impute=True):
    columns = df.columns
    drop_column = []
    impute_column = []
    # 1. Make sure missing doesn't appear in target
    df = df.dropna(axis=0, subset=[target])

    for i, column in enumerate(columns):
        prec = sum(df[column].value_counts()) / len(df)
        if prec < thresh:
            drop_column.append(column)
        elif prec < 1.0:
            impute_column.append(column)
    df = df.drop(drop_column, axis=1)

    if impute:
        '''
        TODO: Configure imputer rather than simply applying SimpleImputer
        '''
        # 1 define transformer of numberical data #(Firstly handle the missing value and then secondly encoding the )
        numerical_transformer = Pipeline(steps=[
            ("Imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())]
        )

        # 3 define transformer of categorical data #(Firstly handle the missing value and then secondly encoding the )
        categorical_transformer = Pipeline(steps=[
            ("Imputer", SimpleImputer(strategy="most_frequent")),
            ("Onehot", OneHotEncoder(handle_unknown='ignore'))]
        )
    else:
        df = df.dropna(axis=0)
        numerical_transformer = Pipeline(steps=[
            ("scaler", StandardScaler())]
        )
        categorical_transformer = Pipeline(steps=[
            ("Onehot", OneHotEncoder(handle_unknown='ignore'))]
        )
    return df, numerical_transformer, categorical_transformer


class Preprocess:
    def __init__(self, file, target,
                 impute=False,
                 missing_threshold=0.25,
                 test_size=0.2,
                 val_size=0.25,
                 num_features=None,
                 cate_features=None,
                 seed=43):
        assert file.endswith("csv") or file.endswith("txt"), "Not a required data type"
        self.df = read_from_csv(file)
        assert target in self.df, "Target column not in given dataframe, please check!"
        self.df, self.num_handler, self.cat_handler = missing_handler(self.df, target, missing_threshold, impute=impute)
        self.y = self.df[target]
        self.X = self.df.drop([target], axis=1)
        self.dev_x, self.test_x, self.dev_y, self.test_y = train_test_split(self.X, self.y, test_size=test_size,
                                                                            random_state=seed)
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(self.dev_x, self.dev_y,
                                                                              test_size=val_size, random_state=seed)

        def make_pipeline(num_f=None, cate_f=None):
            t = self.X.dtypes
            columns = list(t.index)
            if not cate_f:
                cate_f = list(t[(t == "object") | (t == "bool")].index)
            if not num_f:
                num_f = list(filter(lambda x: x not in cate_f, columns))
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", self.num_handler, num_f),
                    ("cat", self.cat_handler, cate_f)]
            )
            return preprocessor

        self.preprocessor = make_pipeline(num_features, cate_features)

    def __fit__(self, X=None, y=None):
        if not X or not y:
            self.preprocessor.fit(self.train_x, self.train_y)
        elif X and y:
            self.preprocessor.fit(X, y)

    def transform(self, X=None, y=None):
        df = self.preprocessor.transform(X)
        print(df)
        return df