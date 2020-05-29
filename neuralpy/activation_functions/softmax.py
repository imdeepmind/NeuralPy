"""Softmax Activation Function"""

from torch.nn import Softmax as _Softmax
from .utils import get_activation_details

class Softmax:
    """
        Applies the Softmax function to the input Tensor rescaling input to the range [0,1].

        Supported Arguments
            dim=None: (Interger) A dimension along which Softmax will be
                computed (so every slice along dim will sum to 1).
            name=None: (String) Name of the activation function layer,
                if not provided then automatically calculates a unique name for the layer.
    """
    def __init__(self, dim=None, name=None):
        """
            __init__ method for the Sigmoid Activation Function class

            Supported Arguments
                dim=None: (Interger) A dimension along which Softmax will be
                    computed (so every slice along dim will sum to 1).
                name=None: (String) Name of the activation function layer,
                    if not provided then automatically calculates a unique name for the layer.
        """
        # Checking the name field, this is an optional field,
        # if not provided generates a unique name for the activation function
        if name is not None:
            if isinstance(name, str):
                if len(name) <= 0:
                    raise ValueError("Please provide a valid name")
            else:
                raise ValueError("Please provide a valid name")

        self.__dim = dim
        self.__name = name

    # pylint: disable=no-self-use,unused-argument
    def get_input_dim(self, prev_input_dim):
        """
            This method calculates the input shape for layer based on previous output layer.
            Here for this activation function, we dont need it

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # Softmax does not need to n_input, so returning None
        return None

    def get_layer(self):
        """
            Provides details of the layer

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # Returning all the details of the activation function
        return get_activation_details(None, None, self.__name, 'Softmax', _Softmax, {
            'dim': self.__dim
        })
