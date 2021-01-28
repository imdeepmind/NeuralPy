"""AlphaDropout Layer"""

from torch.nn import AlphaDropout as _AlphaDropout
from neuralpy.utils import CustomLayer


class AlphaDropout(CustomLayer):
    """
    Applies AlphaDropout to the input.

    The AlphaDropout layer randomly sets input units to 0 with a frequency of
    `rate` at each step during training time, which helps prevent overfitting.
    Inputs not set to 0 are scaled up by 1/(1 - rate)
    such that the sum over all inputs is unchanged.

    Supported Arguments
        p=0.5: (Float) Probability of an element to be zeroed.
                The value should be between 0.0 and 1.0.
        name=None: (String) Name of the layer, if not provided
                    then automatically calculates a unique name for the layer
    """

    def __init__(self, p=0.5, name=None):
        """
        __init__ method for AlphaDropout class

        Supported Arguments
            p=0.5: (Float) Probability of an element to be zeroed.
                    The value should be between 0.0 and 1.0.
            name=None: (String) Name of the layer, if not provided
                        then automatically calculates a unique name for the layer
        """
        super().__init__(_AlphaDropout, "AlphaDropout", layer_name=name)

        # Checking the field p, it should be a valid prob value
        if not (isinstance(p, float) and p >= 0.0 and p <= 1.0):
            raise ValueError("Please provide a valid p value")

        self.__p = p

    def set_input_dim(self, prev_input_dim, layer_type):
        """
        This method calculates the input shape for layer based on
        previous output layer.

        This method is used by the NeuralPy Models, for building the models.
        No need to call this method for using NeuralPy.
        """
        # AlphaDropout does not need to n_input, so returning None
        return None

    def get_layer(self):
        """
        This method returns the details as dict of the layer.

        This method is used by the NeuralPy Models, for building the models.
        No need to call this method for using NeuralPy.
        """
        # Returning all the details of the layer
        return self._get_layer_details(None, {"p": self.__p, "inplace": False})
