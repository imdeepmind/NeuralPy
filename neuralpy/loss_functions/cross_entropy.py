import torch
from torch.nn import CrossEntropyLoss as _CrossEntrypyLoss

# pylint: disable=too-few-public-methods
class CrossEntropyLoss:
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        if weight is not None and not weight:
            raise ValueError("Invalid weight")

        if not isinstance(reduction, str) and reduction not in ["none", "mean", "sum"]:
            raise ValueError("Invalid reduction")

        if not isinstance(ignore_index, int):
            raise ValueError("Invalid ignore index")

        self.__weight = weight
        self.__ignore_index = ignore_index
        self.__reduction = reduction

    def get_loss_function(self):
        return {
            'loss_function': _CrossEntrypyLoss,
            'keyword_arguments': {
                # pylint: disable=not-callable
                'weight': torch.tensor(self.__weight).float(),
                'ignore_index': self.__ignore_index,
                'reduction': self.__reduction
            }
        }
