"""Softmax Activation Function"""

from torch.nn import Softmax as _Softmax
from neuralpy.utils import CustomLayer


class Softmax(CustomLayer):
    """
    Applies the Softmax function to the input Tensor rescaling input to the range
    [0,1].

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
              if not provided then automatically calculates a unique name for the
              layer.
        """
        super().__init__(_Softmax, "Softmax", layer_name=name)

        if not isinstance(dim, int):
            raise ValueError("Please provide a valid dim")

        self.__dim = dim

    def set_input_dim(self, prev_input_dim, layer_type):
        """
        This method calculates the input shape for layer based on previous output
        layer. Here for this activation function, we don't need it

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
        return self._get_layer_details(None, {"dim": self.__dim})
