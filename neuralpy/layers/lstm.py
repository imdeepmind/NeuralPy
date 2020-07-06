""" LSTM layer for NeuralPy"""

from torch.nn import LSTM as _LSTM


class LSTM:
    """
        LSTM Applies a multi-layer long short-term memory(LSTM) RNN
        to an input sequence
        To learn more about LSTM, please check pytorch
        documentation at https://pytorch.org/docs/stable/nn.html#lstm

        Supported Arguments:
            input_size: (Integer) The number of expected features
                in the input
            hidden)size: (Integer) The number of features
                in the hidden state
            num_layers: (Integer) Number of recurrent layers
            bias: (Boolean) If true then uses the bias,
                Defaults to `true`
            batch_first: (Boolean) If `true`, then the
                input and output tensors are provided as
                (batch, seq, feature). Default: `false`
            dropout: (Integer) f non-zero, introduces a
                Dropout layer on the outputs of each LSTM layer
                except the last layer,with dropout probability
                equal to dropout. Default: 0
            bidirectional: (Boolean) If `true`, becomes a
                bidirectional LSTM. Default: `false`


    """

    def __init__(
            self, input_size, hidden_size, num_layers=1,
            bias=True, batch_first=False,dropout=0,
            bidirectional=False, name=None
            ):
        """
            __init__ method for LSTM

            Supported Arguments:
                input_size: (Integer) The number of expected features
                    in the input
                hidden)size: (Integer) The number of features
                    in the hidden state
                num_layers: (Integer) Number of recurrent layers
                bias: (Boolean) If true then uses the bias,
                    Defaults to `true`
                batch_first: (Boolean) If `true`, then the
                    input and output tensors are provided as
                    (batch, seq, feature). Default: `false`
                dropout: (Integer) f non-zero, introduces a
                    Dropout layer on the outputs of each LSTM layer
                    except the last layer,with dropout probability
                    equal to dropout. Default: 0
                bidirectional: (Boolean) If `true`, becomes a
                    bidirectional LSTM. Default: `false`

        """

        # checking the input_size, it is a optional field
        if not input_size or not isinstance(
                input_size, int) and input_size <= 0:
                raise ValueError("Please provide a valid input_size")

        # checking the hidden_size
        if not hidden_size or not isinstance(
                hidden_size, int) and hidden_size <= 0:
                raise ValueError("Please provide a valid hidden_size")

        # checking the num_layers
        if not num_layers or not isinstance(
                num_layers, int) or num_layers <= 0:
                raise ValueError("Please provide a valid num_layers")

        # checking bias, it is an optional field
        if not isinstance(bias, bool):
            raise ValueError("Please provide a valid bias")

        # checking batch_first, it is an optional field
        if not isinstance(batch_first, bool):
            raise ValueError("Please provide a valid batch_first")

        # checking the dropout, it is an optional field
        if not dropout or not isinstance(dropout, int) and dropout < 0:
            raise ValueError("Please provide a valid dropout")

        # checking bidirectional, it is an optional field
        if not isinstance(bidirectional, bool):
            raise ValueError("Please provide a valid bidirectonal")

        # checking the name, it is an optional field
        if name is not None and not (isinstance(name, str) and name):
            raise ValueError("Please provide a valid name")

        # Storing the data
        self.__input_size = input_size
        self.__hidden_size = hidden_size
        self.__num_layers = num_layers

        self.__bias = bias
        self.__batch_first = batch_first
        self.__dropout = dropout
        self.__bidirectional = bidirectional
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
            if layer_type == "lstm":
                self.__input_size = prev_input_dim[-1]
            else:
                raise ValueError("Unsupported previos layer, please provide your own input shape for the layer")

    def get_layer(self):
        """
            This method returns the details as dict of the layer.

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # Returning all the details of the layer
        return{
            'layer_details': self.__num_layers,
            'name': self.__name,
            'type': 'LSTM',
            'layer': _LSTM,
            "keyword_arguments": {
                    'input_size': self.__input_size,
                    'hidden_size': self.__hidden_size,
                    'num_layers': self.__num_layers,
                    'bias': self.__bias,
                    'batch_first': self.__batch_first,
                    'dropout': self.__dropout,
                    'bidirectional': self.__bidirectional
            }
        }
