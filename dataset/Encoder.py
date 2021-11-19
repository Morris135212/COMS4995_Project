from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime


class DateEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, format_str):
        super(DateEncoder, self).__init__()
        self.format = format_str

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        pass


class DateHourEncoder(DateEncoder):
    def __init__(self, format_str):
        super(DateHourEncoder, self).__init__(format_str)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        hour = X.map(lambda x: datetime.strptime(x, self.format).hour)

        return hour


class DateWeekEncoder(DateEncoder):
    def __init__(self, format_str):
        super(DateWeekEncoder, self).__init__(format_str)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        weekday = X.map(lambda x: datetime.strptime(x, self.format).weekday())
        return weekday


class DateMonthEncoder(DateEncoder):
    def __init__(self, format_str):
        super(DateMonthEncoder, self).__init__(format_str)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        month = X.map(lambda x:datetime.strptime(x, self.format).month)
        return month


class DateYearEncoder(DateEncoder):
    def __init__(self, format_str):
        super(DateYearEncoder, self).__init__(format_str)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        year = X.map(lambda x: datetime.strptime(x, self.format).year)
        return year
