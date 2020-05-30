"""RMSProps Optimizer"""

from torch.optim import RMSprop as _RMSprop

# pylint: disable=too-few-public-methods
class RMSprop:
    # pylint: disable=too-many-arguments
    def __init__(self, learning_rate=0.001, alpha=0.99, eps=1e-08, weight_decay=0,
                 momentum=0, centered=False):
        if not isinstance(learning_rate, float) or learning_rate < 0.0:
            raise ValueError("Invalid learning_rate")

        if not isinstance(eps, float) or eps < 0.0:
            raise ValueError("Invalid eps value")

        if not isinstance(weight_decay, float) or weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value")

        self.__learning_rate = learning_rate
        self.__alpha = alpha
        self.__eps = eps
        self.__weight_decay = weight_decay
        self.__momentum = momentum
        self.__centered = centered

    def get_optimizer(self):
        """
            Method used for getting the details of the optimizer

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        return {
            'optimizer': _RMSprop,
            'keyword_arguments': {
                'lr': self.__learning_rate,
                'alpha': self.__alpha,
                'eps': self.__eps,
                'weight_decay': self.__weight_decay,
                'momentum': self.__momentum,
                'centered': self.__centered
            }
        }
