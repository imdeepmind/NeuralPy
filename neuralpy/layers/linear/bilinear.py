"""Bilinear layer for NeuralPy"""

from torch.nn import Bilinear as _BiLinear
from neuralpy.utils import CustomLayer


class Bilinear(CustomLayer):
    """
    A bilinear layer is a function of two inputs x and y
    that is linear in each input separately.
    Simple bilinear functions on vectors are the
    dot product or the element-wise product.

    To learn more about Dense layers, please check PyTorch
    documentation at https://pytorch.org/docs/stable/nn.html#bilinear

    Supported Arguments:
    n_nodes: (Integer) Size of the output sample
    n1_features: (Integer) Size of the input sample1,
        no need for this argument layers except the initial layer.
    n2_features: (Integer) Size of the input sample2,
        no need for this argument layers except the initial layer
    bias: (Boolean) If true then uses the bias, Defaults to `true`
    name: (String) Name of the layer, if not provided then
        automatically calculates a unique name for the layer
    """

    def __init__(
        self, n_nodes, n1_features=None, n2_features=None, bias=True, name=None
    ):
        """
        __init__ method for bilinear layer

        Supported Arguments:
        n_nodes: (Integer) Size of the output sample
        n1_features: (Integer) Size of the input sample1,
            no need for this argument layers except the initial layer.
        n2_features: (Integer) Size of the input sample2,
            no need for this argument layers except the initial layer
        bias: (Boolean) If true then uses the bias, Defaults to `true`
        name: (String) Name of the layer, if not provided then
            automatically calculates a unique name for the layer

        """
        # Checking the n_nodes field
        if not n_nodes or not isinstance(n_nodes, int) or n_nodes <= 0:
            raise ValueError("Please provide a valid n_nodes")

        # Checking the n1_features field, it is a optional field
        if n1_features is not None and not (
            isinstance(n1_features, int) and n1_features >= 0
        ):
            raise ValueError("Please provide a valid n1_features")

        # Checking the n2_features field
        if n2_features is not None and not (
            isinstance(n2_features, int) and n2_features >= 0
        ):
            raise ValueError("Please provide a valid n2_features")

        # Checking the bias field, this is also optional, default to True
        if not isinstance(bias, bool):
            raise ValueError("Please provide a valid bias")

        super().__init__(_BiLinear, "Bilinear", layer_name=name)

        # Storing the data
        self.__n_inputs = n1_features
        self.__n_inputs2 = n2_features
        self.__n_nodes = n_nodes

        self.__bias = bias

    def set_input_dim(self, prev_input_dim, prev_layer_type):
        """
        This method calculates the input shape for layer based on previous output
        layer.

        This method is used by the NeuralPy Models, for building the models.
        No need to call this method for using NeuralPy.
        """
        # Checking if n_inputs is there or not, not overwriting the n_input
        # field
        if not self.__n_inputs:
            layer_type = prev_layer_type.lower()

            if layer_type in ("dense", "batchnorm1d", "bilinear"):
                self.__n_inputs = prev_input_dim[0]
            else:
                raise ValueError(
                    "Unsupported previous layer, please provide your own input shape \
                    for the layer"
                )

    def get_layer(self):
        """
        This method returns the details as dict of the layer.

        This method is used by the NeuralPy Models, for building the models.
        No need to call this method for using NeuralPy.
        """
        # Returning all the details of the layer
        return self._get_layer_details(
            (self.__n_nodes,),
            {
                "in_features1": self.__n_inputs,
                "in_features2": self.__n_inputs2,
                "out_features": self.__n_nodes,
                "bias": self.__bias,
            },
        )
