"""Stochastic Gradient Descent"""

from torch.optim import SGD as _SGD

# pylint: disable=too-few-public-methods
class SGD:
    """
        Applies a SGD (Stochastic Gradient Descent) with momentum.

        Supported Arguments
            learning_rate=0.001: (Float) Learning Rate for the optimizer
            momentum=0 : (Float) Momentum for the optimizer
            dampening=0 : (Float) Dampening of momentum
            weight_decay=0 : (Float) Weight decay for the optimizer
            nesterov=False : (Bool) Enables Nesterov momentum
    """
    # pylint: disable=too-many-arguments
    def __init__(self, learning_rate=0.001, momentum=0.0,
                 dampening=0.0, weight_decay=0.0, nesterov=False):

        """
            __init__ method for SGD

            Supported Arguments
                learning_rate=0.001: (Float) Learning Rate for the optimizer
                momentum=0 : (Float) Momentum for the optimizer
                dampening=0 : (Float) Dampening of momentum
                weight_decay=0 : (Float) Weight decay for the optimizer
                nesterov=False : (Bool) Enables Nesterov momentum
        """

        # Validation the input fields
        if not isinstance(learning_rate, float) or learning_rate < 0.0:
            raise ValueError("Invalid learning_rate")

        if not isinstance(momentum, float) or momentum < 0.0:
            raise ValueError("Invalid momentum value")

        if not isinstance(weight_decay, float) or weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value")

        if not isinstance(dampening, float):
            raise ValueError("Invalid dampening parameter")

        if not isinstance(nesterov, bool):
            raise ValueError("Invalid nesterov parameter")

        self.__learning_rate = learning_rate
        self.__momentum = momentum
        self.__dampening = dampening
        self.__weight_decay = weight_decay
        self.__nesterov = nesterov

    def get_optimizer(self):
        """
            Method used for getting the details of the optimizer

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # Returning the optimizer data
        return {
            'optimizer': _SGD,
            'keyword_arguments': {
                'lr': self.__learning_rate,
                'momentum': self.__momentum,
                'dampening': self.__dampening,
                'weight_decay': self.__weight_decay,
                'nesterov': self.__nesterov
            }
        }
