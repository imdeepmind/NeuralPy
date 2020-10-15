"""GRUCell layer for NeuralPy"""


from torch.nn import GRUCell as _GRUCell


class GRUCell:
    """
        A gated recurrent unit (GRU) cell
        To learn more about RNN, please check pytorch
        documentation at https://pytorch.org/docs/stable/nn.html#grucell

        Supported Arguments:
            input_size: (Integer) The number of expected features
                in the input
            hidden)size: (Integer) The number of features
                in the hidden state
            bias: (Boolean) If true then uses the bias,
                Defaults to `true`
            name: (String) Name of the layer, if not provided then
                automatically calculates a unique name for the layer
    """

    def __init__(
            self, input_size, hidden_size, bias=True, name=None
        ):

        """
            __init__ method for GRUCell

            Supported Arguments:
            input_size: (Integer) The number of expected features
                in the input
            hidden)size: (Integer) The number of features
                in the hidden state
            bias: (Boolean) If true then uses the bias,
                Defaults to `true`
            name: (String) Name of the layer, if not provided then
                automatically calculates a unique name for the layer
        """

        if input_size is not None and not (isinstance(
                input_size, int) and input_size > 0):
            raise ValueError("Please provide a valid input_size")

        if not hidden_size or not isinstance(
                hidden_size, int) or hidden_size <= 0:
            raise ValueError("Please provide a valid hidden_size")

        if not isinstance(bias, bool):
            raise ValueError("Please provide a valid bias")

        if name is not None and not (isinstance(name, str) and name):
            raise ValueError("Please provide a valid name")

        self.__input_size = input_size
        self.__hidden_size = hidden_size

        self.__bias = bias
        self.__name = name


    def get_input_dim(self, prev_input_dim, prev_layer_type):
        """
            This method calculates the input shape for layer based on previous output layer.

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
       # Checking if n_inputs is there or not, not overwriting the n_input field
        if not self.__input_size:
            layer_type = prev_layer_type.lower()

            # based on the prev layer type, predicting the n_inputs
            # to support more layers, we need to add some more statements
            if layer_type in ("grucell", "rnn", "lstm", "gru", "dense", "embedding"):
                self.__input_size = prev_input_dim[-1]
            else:
                raise ValueError(
                    "Unsupported previous layer, please provide your own input shape for the layer")

    def get_layer(self):
        """
            This method returns the details as dict of the layer.

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # Returning all the details of the layer
        return{
            'layer_details': (self.__hidden_size, ),
            'name': self.__name,
            'type': 'GRUCell',
            'layer': _GRUCell,
            'keyword_arguments': {
                'input_size': self.__input_size,
                'hidden_size': self.__hidden_size,
                'bias': self.__bias
            }
        }
