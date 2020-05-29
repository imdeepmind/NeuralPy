"""LeakyReLU Activation Function"""

from torch.nn import LeakyReLU as _LeakyReLU
from .utils import get_activation_details


class LeakyReLU:
    """
        LeakyReLU is a modified ReLU activation function with some improvements.
        LeakyReLU solves the problem of "dead ReLU", by introducing a new parameter
        called the negative slope.

        In traditional ReLU, if the input is negative, then the output is 0.
        But for LeakyReLU, the output is not zero. This feature special behavior
        of LeakyReLU solves the problem of "dead ReLU" and helps in learning.

        Supported Arguments
                negative_slope=0.01: (Integer) A negative slope for the LeakyReLU
                name=None: (String) Name of the activation function layer, if not
                        provided then automatically calculates a unique name for the layer
    """

    def __init__(self, negative_slope=0.01, name=None):
        """
            __init__ method for the LeakyReLU Activation Function class

            Supported Arguments
                        negative_slope=0.01: (Integer) A negative slope for the LeakyReLU
                        name=None: (String) Name of the activation function layer, if not
                                provided then automatically calculates a unique name for the layer
        """
        # Checking the name field, this is an optional field,
        # if not provided generates a unique name for the activation function
        if name is not None:
            if isinstance(name, str):
                if len(name) <= 0:
                    raise ValueError("Please provide a valid name")
            else:
                raise ValueError("Please provide a valid name")

        self.__negative_slope = negative_slope
        self.__name = name

    # pylint: disable=no-self-use,unused-argument
    def get_input_dim(self, prev_input_dim):
        """
            This method calculates the input shape for layer based on previous output layer.
            Here for this activation function, we dont need it

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # LeakyReLU does not need to n_input, so returning None
        return None

    def get_layer(self):
        """
            Provides details of the layer

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # Returning all the details of the activation function
        return get_activation_details(None, None, self.__name, 'LeakyReLU', _LeakyReLU, {
            'negative_slope': self.__negative_slope,
            'inplace': False
        })
