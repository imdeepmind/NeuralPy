"""LeakyReLU Activation Function"""

from torch.nn import LeakyReLU as _LeakyReLU
from neuralpy.utils import CustomLayer


class LeakyReLU(CustomLayer):
    """
    LeakyReLU is a modified ReLU activation function with some improvements.
    LeakyReLU solves the problem of "dead ReLU", by introducing a new parameter
    called the negative slope.

    In traditional ReLU, if the input is negative, then the output is 0.
    But for LeakyReLU, the output is not zero. This feature special behavior
    of LeakyReLU solves the problem of "dead ReLU" and helps in learning.

    Supported Arguments
      negative_slope=0.01: (Float) A negative slope for the LeakyReLU
      name=None: (String) Name of the activation function layer, if not
            provided then automatically calculates a unique name for the layer
    """

    def __init__(self, negative_slope=0.01, name=None):
        """
        __init__ method for the LeakyReLU Activation Function class

        Supported Arguments
          negative_slope=0.01: (Float) A negative slope for the LeakyReLU
          name=None: (String) Name of the activation function layer, if not
                provided then automatically calculates a unique name for the layer
        """
        super().__init__(_LeakyReLU, "LeakyReLU", layer_name=name)

        if not isinstance(negative_slope, float):
            raise ValueError("Please provide a valid negative slope")

        self.__negative_slope = negative_slope

    def set_input_dim(self, prev_input_dim, layer_type):
        """
        This method calculates the input shape for layer based on previous output
        layer. Here for this activation function, we don't need it

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
        return self._get_layer_details(
            None, {"negative_slope": self.__negative_slope, "inplace": False}
        )
