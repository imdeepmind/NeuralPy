"""RNNCell layer for NeuralPy"""


from torch.nn import RNNCell as _RNNCell


class RNNCell:

    def __init__(
            self, input_size, hidden_size,
            bias=True, non_linearity='tanh', name=None
    ):

        if input_size is not None and not (isinstance(
                input_size, int) and input_size > 0):
            raise ValueError("Please provide a valid input_size")

        if not hidden_size or not isinstance(
                hidden_size, int) or hidden_size <= 0:
            raise ValueError("Please provide a valid hidden_size")
        
        if not isinstance(bias, bool):
            raise ValueError("Please provide a valid bias")

        if non_linearity not in ("tanh", "relu"):
            raise ValueError("Please provide a valid non_linearity")

        if name is not None and not (isinstance(name, str) and name):
            raise ValueError("Please provide a valid name")
        
        self.__input_size = input_size
        self.__hidden_size = hidden_size

        self.__bias  = bias
        self.__non_linearity = non_linearity
        self.__name = name


    def get_input_dim(self, prev_input_dim, prev_layer_type):
        """
            This method calculates the input shape for layer based on previous output layer.

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        return None

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
            'type': 'RNNCell',
            'layer': _RNNCell,
            'keyword_arguments': {
                    'input_size': self.__input_size,
                    'hidden_size': self.__hidden_size,
                    'bias': self.__bias,
                    'non_linearity': self.__non_linearity
            }
        }

