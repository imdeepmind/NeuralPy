"""Sigmoid Activation Function"""

from torch.nn import Sigmoid as _Sigmoid
from neuralpy.utils import CustomLayer


class Sigmoid(CustomLayer):
    """
    Applies a element-wise Sigmoid or Logistic function to the input tensor.

    Supported Arguments
      name=None: (String) Name of the activation function layer,
          if not provided then automatically calculates a unique name for the layer.
    """

    def __init__(self, name=None):
        """
        __init__ method for the Sigmoid Activation Function class

        Supported Arguments
          name=None: (String) Name of the activation function layer,
              if not provided then automatically calculates a unique name for the
              layer
        """
        super().__init__(_Sigmoid, "Sigmoid", layer_name=name)

    def set_input_dim(self, prev_input_dim, layer_type):
        """
        This method calculates the input shape for layer based on previous output
        layer. Here for this activation function, we don't need it

        This method is used by the NeuralPy Models, for building the models.
        No need to call this method for using NeuralPy.
        """
        # Sigmoid does not need to n_input, so returning None
        return None

    def get_layer(self):
        """
        Provides details of the layer

        This method is used by the NeuralPy Models, for building the models.
        No need to call this method for using NeuralPy.
        """
        # Returning all the details of the activation function
        return self._get_layer_details()
