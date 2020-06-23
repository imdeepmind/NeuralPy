""" Rprop Optimizer"""

from torch.optim import Rprop as _Rprop

class Rprop:
    """
    Implements Rprop oprimizer

    Supported Arguments
        learning_rate=0.01: (Float) Learning Rate for the optimizer
        etas=(0.5,1.2): (Tuple) pair of (etaminus, etaplis), that are
            multiplicative increase and decrease factors
        step_sizes=(le-06,50): (Tuple) a pair of
            minimal and maximal allowed step sizes for the optimizer

    """
    def __init__(
            self, learning_rate=0.01, etas=(0.5, 1.2),
            step_sizes=(1e-06, 50)):
        """
            __init__ method for Rprop

            Supported Arguments
                learning_rate=0.01: (Float) Learning Rate for the optimizer
                etas=(0.5,1.2): (Tuple) pair of (etaminus, etaplis), that are
                    multiplicative increase and decrease factors
                step_sizes=(le-06,50): (Tuple) a pair of minimal and
                    maximal allowed step sizes for the optimizer
        """
        if not isinstance(
                learning_rate, float) or learning_rate <= 0.0:
            raise ValueError("Invalid learning_rate")

        if not isinstance(etas[0], float):
            raise ValueError("Invalid etas parameter at index 0")

        if not isinstance(etas[1], float):
            raise ValueError("Invalid etas parameter at index 1")

        if not isinstance(step_sizes[0], float):
            raise ValueError("Invalid step_size parameter at index 0")

        if not isinstance(step_sizes[1], float):
            raise ValueError("Invalid step_size parameter at index 1")

        self.__learning_rate = learning_rate
        self.__etas = etas
        self.__step_sizes = step_sizes

    def get_optimizer(self):
        """
            Method used for getting the details of the optimizer

            This method is used by the NeuralPy Models,
            for building the models.
            No need to call this method for using NeuralPy.
        """
        return{
            'optimizer': _Rprop,
            'keyword_arguments': {
                'lr': self.__learning_rate,
                'etas': self.__etas,
                'step_sizes': self.__step_sizes
            }
        }
