"""Dense layer for NeuralPy"""

from torch.nn import Linear as _Dense
from neuralpy.utils import CustomLayer


class Dense(CustomLayer):
    """
    A Dense is a normal densely connected Neural Network.
    It performs a linear transformation of the input.

    To learn more about Dense layers, please check PyTorch
    documentation at https://pytorch.org/docs/stable/nn.html?highlight=linear

    Supported Arguments:
        n_nodes: (Integer) Size of the output sample
        n_inputs: (Integer) Size of the input sample,
            no need for this argument layers except the initial layer.
        bias: (Boolean) If true then uses the bias, Defaults to `true`
        name: (String) Name of the layer, if not provided then
            automatically calculates a unique name for the layer
    """

    def __init__(self, n_nodes, n_inputs=None, bias=True, name=None):
        """
        __init__ method for the Dense layer

        Supported Arguments:
            n_nodes: (Integer) Size of the output sample
            n_inputs: (Integer) Size of the input sample,
                no need for this argument layers except the initial layer.
            bias: (Boolean) If true then uses the bias, Defaults to `true`
            name: (String) Name of the layer, if not provided then
                automatically calculates a unique name for the layer
        """
        # Checking the n_nodes field
        if not n_nodes or not isinstance(n_nodes, int) or n_nodes <= 0:
            raise ValueError("Please provide a valid n_nodes")

        # Checking the n_input field, it is a optional field
        if n_inputs is not None and not (isinstance(n_inputs, int) and n_inputs >= 0):
            raise ValueError("Please provide a valid n_inputs")

        # Checking the bias field, this is also optional, default to True
        if not isinstance(bias, bool):
            raise ValueError("Please provide a valid bias")

        super().__init__(_Dense, "Dense", layer_name=name)

        # Storing the data
        self.__n_inputs = n_inputs
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

            # based on the prev layer type, predicting the n_inputs
            # to support more layers, we need to add some more statements
            if layer_type == "dense":
                self.__n_inputs = prev_input_dim[0]
            elif layer_type in ("rnn", "lstm", "gru", "rnncell", "lstmcell", "grucell"):
                self.__n_inputs = prev_input_dim[-1]
            elif layer_type in (
                "conv1d",
                "conv2d",
                "conv3d",
                "avgpool1d",
                "avgpool2d",
                "avgpool3d",
                "maxpool1d",
                "maxpool2d",
                "maxpool3d",
                "batchnorm1d",
                "batchnorm2d",
                "batchnorm3d",
            ):
                self.__n_inputs = prev_input_dim[1]
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
                "in_features": self.__n_inputs,
                "out_features": self.__n_nodes,
                "bias": self.__bias,
            },
        )
