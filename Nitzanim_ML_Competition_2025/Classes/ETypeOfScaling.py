from enum import Enum

class ETypeOfScaling(Enum):
    """
    An enum which contains the type of scaling
    """
    NO_SCALING = 1
    NORMALIZATION = 2
    STANDARDIZATION = 3