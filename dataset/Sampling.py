from enum import Enum
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE


class SampleMechanism(Enum):
    under = "undersampling"
    over = "oversampling"
    SMOTE = "smote"


class Sampling:
    def __init__(self, X, y, mechanism=SampleMechanism.under, seed=23):
        if mechanism == SampleMechanism.under:
            self.sample = RandomUnderSampler(replacement=False)
        elif mechanism == SampleMechanism.over:
            self.sample = RandomOverSampler()
        elif mechanism == SampleMechanism.SMOTE:
            self.sample = SMOTE(random_state=seed)
        else:
            raise RuntimeError("Not a required sampling mechanism")
        self.X_resample, self.y_resample = self.sample.fit_resample(X, y)

    def get_features(self):
        return self.X_resample

    def get_labels(self):
        return self.y_resample
