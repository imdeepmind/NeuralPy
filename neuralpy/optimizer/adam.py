"""Adam Optimizer"""

from torch.optim import Adam as _Adam

# pylint: disable=too-few-public-methods
class Adam:
    # pylint: disable=too-many-arguments
    def __init__(self, learning_rate=0.001, betas=(0.9, 0.999), eps=1e-08, 
                 weight_decay=0, amsgrad=False):
        if not isinstance(learning_rate, float) or learning_rate < 0.0:
            raise ValueError("Invalid learning_rate")

        if not isinstance(eps, float) or eps < 0.0:
            raise ValueError("Invalid eps value")

        if not isinstance(betas[0], float) or betas[0] < 0.0:
            raise ValueError("Invalid beta parameter at index 0")

        if not isinstance(betas[1], float) or betas[1] < 0.0:
            raise ValueError("Invalid beta parameter at index 1")

        if not isinstance(weight_decay, float) or weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value")

        self.__learning_rate = learning_rate
        self.__betas = betas
        self.__eps = eps
        self.__weight_decay = weight_decay
        self.__amsgrad = amsgrad

    def get_optimizer(self):
        """
            Method used for getting the details of the optimizer

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        return {
            'optimizer': _Adam,
            'keyword_arguments': {
                'lr': self.__learning_rate,
                'betas': self.__betas,
                'eps': self.__eps,
                'weight_decay': self.__weight_decay,
                'amsgrad': self.__amsgrad
            }
        }
