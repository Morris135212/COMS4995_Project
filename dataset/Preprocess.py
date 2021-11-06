from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


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
        numerical_transformer = [("Impute", SimpleImputer(strategy="mean"))]

        # 3 define transformer of categorical data #(Firstly handle the missing value and then secondly encoding the )
        categorical_transformer = [("Impute", SimpleImputer(strategy="most_frequent"))]
    else:
        df = df.dropna(axis=0)
        numerical_transformer = []
        categorical_transformer = []
    return df, numerical_transformer, categorical_transformer


class Preprocess:
    def __init__(self, df, target,
                 impute=False,
                 missing_threshold=0.25,
                 num_features=None,
                 cate_features=None):

        self.df = df
        assert target in self.df, "Target column not in given dataframe, please check!"
        self.df, self.num_handler, self.cat_handler = missing_handler(self.df, target, missing_threshold, impute=impute)
        self.tar_handler = LabelEncoder()
        self.y = self.df[target]
        self.X = self.df.drop([target], axis=1)

        # self.dev_x, self.test_x, self.dev_y, self.test_y = train_test_split(self.X, self.y, test_size=test_size,
        #                                                                     random_state=seed)
        # self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(self.dev_x, self.dev_y,
        #                                                                       test_size=val_size, random_state=seed)

        def make_pipeline(num_f=None, cate_f=None):
            t = self.X.dtypes
            columns = list(t.index)
            if not cate_f:
                cate_f = list(t[(t == "object") | (t == "bool")].index)
            if not num_f:
                num_f = list(filter(lambda x: x not in cate_f, columns))
            self.num_handler += [("scaler", StandardScaler())]
            self.cat_handler += [("onehot", OneHotEncoder())]
            num_transformer = Pipeline(steps=self.num_handler)
            cate_transformer = Pipeline(steps=self.cat_handler)
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_transformer, num_f),
                    ("cat", cate_transformer, cate_f)
                ]
            )
            return preprocessor

        self.preprocessor = make_pipeline(num_features, cate_features)

    def fit_column(self):
        self.preprocessor.fit(self.X)

    def fit_y(self):
        self.tar_handler.fit(self.y)

    def get_column_pipeline(self):
        return self.preprocessor

    def get_target_pipeline(self):
        return self.tar_handler
