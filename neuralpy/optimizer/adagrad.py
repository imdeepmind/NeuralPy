"""Adagrad Optimizer"""

from torch.optim import Adagrad as _Adagrad

# pylint: disable=too-few-public-methods
class Adagrad:
    """
        Implements Adagrad optimizer.

        Supported Arguments:
            learning_rate=0.001: (Float) Learning Rate for the optimizer
            learning_rate_decay=(0.9,0.999) : (Float) Learning Rate decay
            eps=0 : (Float) Term added to the denominator to improve numerical stability
            weight_decay=0 : (Float) Weight decay for the optimizer
    """
    def __init__(self, learning_rate=0.001, learning_rate_decay=0.0, eps=1e-08, weight_decay=0.0):
        """
            __init__ method for Adagrad optimizer

            Supported Arguments:
                learning_rate=0.001: (Float) Learning Rate for the optimizer
                learning_rate_decay=(0.9,0.999) : (Float) Learning Rate decay
                eps=0 : (Float) Term added to the denominator to improve numerical stability
                weight_decay=0 : (Float) Weight decay for the optimizer
        """
        if not isinstance(learning_rate, float) or learning_rate <= 0.0:
            raise ValueError("Invalid learning_rate")

        if not isinstance(learning_rate_decay, float):
            raise ValueError("Invalid learning_rate_decay parameter")

        if not isinstance(eps, float) or eps < 0.0:
            raise ValueError("Invalid eps value")

        if not isinstance(weight_decay, float) or weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value")

        self.__learning_rate = learning_rate
        self.__learning_rate_decay = learning_rate_decay
        self.__eps = eps
        self.__weight_decay = weight_decay

    def get_optimizer(self):
        """
            Method used for getting the details of the optimizer

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        return {
            'optimizer': _Adagrad,
            'keyword_arguments': {
                'lr': self.__learning_rate,
                'lr_decay': self.__learning_rate_decay,
                'weight_decay': self.__weight_decay,
                # 'eps': self.__eps
            }
        }
