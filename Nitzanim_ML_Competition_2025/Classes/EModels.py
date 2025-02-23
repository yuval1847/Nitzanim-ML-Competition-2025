from enum import Enum

class EModels(Enum):
    """
    An enum which contains the supervised models names
    """
    LOGISTIC_REGRESSION = 1
    RANDOM_FOREST = 2
    KNN = 3
    GAUSSIAN_NAIVE_BAYES = 4
    SVM = 5