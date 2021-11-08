from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


class MissingHandler:
    def __init__(self, df, target,
                 thresh=0.25, impute=True, num_imputer="SimpleImputer", cate_imputer="SimpleImputer"):
        self.df = df
        self.columns = df.columns
        self.target = target
        self.thresh = thresh
        self.impute = impute
        self.imputer = {"num":
                            {"SimpleImputer": ("Impute", SimpleImputer(strategy="mean"))},
                        "cate":
                            {"SimpleImputer": ("Impute", SimpleImputer(strategy="most_frequent"))}
                        }

        def missing_handler(n_imputer="SimpleImputer", c_imputer="SimpleImputer"):
            drop_column = []
            impute_column = []
            # 1. Make sure missing doesn't appear in target
            df = self.df.dropna(axis=0, subset=[target])
            for i, column in enumerate(self.columns):
                prec = sum(df[column].value_counts()) / len(df)
                if prec < self.thresh:
                    drop_column.append(column)
                elif prec < 1.0:
                    impute_column.append(column)
            df = self.df.drop(drop_column, axis=1)
            if self.impute:
                # 1 define transformer of numberical data
                n_transformer = [self.imputer["num"][n_imputer]]
                # 2 define transformer of categorical data
                c_transformer = [self.imputer["cate"][c_imputer]]
            else:
                df = df.dropna(axis=0)
                n_transformer = []
                c_transformer = []
            return df, n_transformer, c_transformer

        self.df, self.num_transformer, self.cate_transformer = missing_handler(num_imputer, cate_imputer)

    def get_dataframe(self):
        return self.df

    def get_features(self):
        return self.df.drop([self.target], axis=1)

    def get_labels(self):
        return self.df[self.target]


class Preprocess:
    def __init__(self, X, y, missing_handler,
                 num_features=None,
                 cate_features=None):

        self.tar_handler = LabelEncoder()
        self.y = y
        self.X = X
        self.missing_handler = missing_handler

        def make_pipeline(num_f=None, cate_f=None):
            t = self.X.dtypes
            columns = list(t.index)
            if not cate_f:
                cate_f = list(t[(t == "object") | (t == "bool")].index)
            if not num_f:
                num_f = list(filter(lambda x: x not in cate_f, columns))
            num_transformer = Pipeline(steps=self.missing_handler.num_transformer +
                                             [("scaler", StandardScaler())])
            oh_cate_transformer = Pipeline(steps=self.missing_handler.cate_transformer +
                                              [("onehot", OneHotEncoder(handle_unknown="ignore"))])
            # tar_cate_transformer = Pipeline(steps=self.missing_handler.cate_transformer +
            #                                   [("onehot", TargetEncoder(handle_unknown="value"))])
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_transformer, num_f),
                    ("cat", oh_cate_transformer, cate_f)
                ]
            )
            return preprocessor

        self.preprocessor = make_pipeline(num_features, cate_features)

    def __fit__(self):
        self.fit_column()
        self.fit_y()

    def fit_column(self):
        self.preprocessor.fit(self.X, self.y)

    def fit_y(self):
        self.tar_handler.fit(self.y)

    def get_column_pipeline(self):
        return self.preprocessor

    def get_target_pipeline(self):
        return self.tar_handler
