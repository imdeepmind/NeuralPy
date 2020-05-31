import torch
from torch.nn import BCEWithLogitsLoss as _BCEWithLogitsLoss

# pylint: disable=too-few-public-methods
class BCELoss:
    def __init__(self, weight=None, reduction='mean', pos_weight=None):
        if weight is not None and not weight:
            raise ValueError("Invalid weight")

        if not isinstance(reduction, str) and reduction not in ["none", "mean", "sum"]:
            raise ValueError("Invalid reduction")

        if pos_weight is not None and not pos_weight:
            raise ValueError("Invalid pos_weight")

        self.__weight = weight
        self.__reduction = reduction
        self.__pos_weight = pos_weight

    def get_loss_function(self):
        return {
            'loss_function': _BCEWithLogitsLoss,
            'keyword_arguments': {
                # pylint: disable=not-callable
                'weight': torch.tensor(self.__weight).float(),
                'reduction': self.__reduction,
                # pylint: disable=not-callable
                'pos_weight': torch.tensor(self.__pos_weight).float()
            }
        }
