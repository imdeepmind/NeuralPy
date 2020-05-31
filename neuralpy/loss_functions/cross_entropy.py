"""CrossEntropyLoss"""

import numpy as np
import torch
from torch.nn import CrossEntropyLoss as _CrossEntrypyLoss

# pylint: disable=too-few-public-methods


class CrossEntropyLoss:
    """
        Applies a Cross Entropy Loss function to the model.

        Cross Entropy Loss automatically applies a Softmax
        Layer at the end of the model, so there is no need to add a Softmax layer.

        Supported Arguments
            weight=None : (Numpy Array | List) Manual rescaling of classes
            ignore_index=-100 : (Integer) Specifies a target value that is
                ignored and does not contribute to the input gradient.
            reduction='mean' : (String) Specifies the reduction that is to be
                applied to the output.
    """

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        """
            __init__ method for CrossEntropyLoss

            Supported Arguments
                weight=None : (Numpy Array | List) Manual rescaling of classes
                ignore_index=-100 : (Integer) Specifies a target value that is
                    ignored and does not contribute to the input gradient.
                reduction='mean' : (String) Specifies the reduction that is to be
                    applied to the output.
        """
        if weight is not None and not (
                isinstance(weight, list) or
                type(weight).__module__ == np.__name__):
            raise ValueError("Invalid weight")

        if reduction not in ["none", "mean", "sum"]:
            raise ValueError("Invalid reduction")

        if not isinstance(ignore_index, int):
            raise ValueError("Invalid ignore index")

        self.__weight = weight
        self.__ignore_index = ignore_index
        self.__reduction = reduction

    def get_loss_function(self):
        """
            Returns the details of the loss function

            There is no need to call this method as this is used by the
            Sequential model to build the model
        """
        return {
            'loss_function': _CrossEntrypyLoss,
            'keyword_arguments': {
                # pylint: disable=not-callable
                'weight': torch.tensor(self.__weight).float(),
                'ignore_index': self.__ignore_index,
                'reduction': self.__reduction
            }
        }
