"""ReLU Activation Function"""

from torch.nn import ReLU as _ReLU
from .utils import get_activation_details, validate_name_field


class ReLU:
    """
        ReLU applies a rectified linear unit function to the input tensors.

        For more information, check https://pytorch.org/docs/stable/nn.html#relu

        Supported Arguments
            name=None: (String) Name of the activation function layer,
                if not provided then automatically calculates a unique name for the layer
    """

    def __init__(self, name=None):
        """
            __init__ method for the ReLU Activation Function class

            Supported Arguments
                name=None: (String) Name of the activation function layer,
                    if not provided then automatically calculates a unique name for the layer
        """
        # Checking the name field, this is an optional field,
        # if not provided generates a unique name for the activation function
        validate_name_field(name)

        self.__name = name

    # pylint: disable=no-self-use,unused-argument
    def get_input_dim(self, prev_input_dim, layer_type):
        """
            This method calculates the input shape for layer based on previous output layer.
            Here for this activation function, we dont need it

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # ReLU does not need to n_input, so returning None
        return None

    def get_layer(self):
        """
            Provides details of the layer

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # Returning all the details of the activation function
        return get_activation_details(self.__name, 'ReLU', _ReLU, {
            'inplace': False
        })
