"""RNN layer for NeuralPy"""

from torch.nn import RNN as _RNN

class RNN:

    def __init__(
            self, input_size, hidden_size, num_layers=1,
            non_linearity='tanh', bias=True, batch_first=False,
            dropout=0, bidirectional=False, name=None 
        ):

        if not input_size or not isinstance(
            input_size, int) and input_size <=0:
                raise ValueError("Please provide a valid input_size")

        if not hidden_size or not isinstance(
            hidden_size, int) and hidden_size <=0:
                raise ValueError("Please provide a valid hidden_size")

        if not isinstance(
            num_layers, int) or num_layers <=0:
                raise ValueError("Please provide a valid num_layers")
        
        if not isinstance(non_linearity, str):
            raise ValueError("Please provide a valid non_linearity")

        if not isinstance(bias, bool):
            raise ValueError("Please provide a valid bias")

        if not isinstance(batch_first, bool):
            raise ValueError("Please provide a valid batch_first")

        if not isinstance(dropout, int) or dropout <0:
            raise ValueError("Please provide a valid dropout")

        if not isinstance(bidirectional, bool):
            raise ValueError("Please provide a valid bidirectonal")

        if name is not None and not (isinstance(name, str) and name):
            raise ValueError("Please provide a valid name")

        self.__input_size = input_size
        self.__hidden_size = hidden_size
        self.__num_layers = num_layers
        
        self.__non_linearity = non_linearity
        self.__bias = bias
        self.__batch_first = batch_fisrt
        self.__dropout = dropout
        self.__bidirectional = bidirectional
        self.__name = name

    def get_input_dim(self, prev_input_dim):

        if not self.__input_size:
            self.__input_size = prev_input_dim

    def get_layer(self):

        return{
            'input_size': self.__input_size,
            'hidden_size': self.__hidden_size,
            'num_layers': self.__num_layers,
            'non_linearity': self.__non_linearity,
            'bias': self.__bias,
            'batch_first': self.__batch_first,
            'dropout': self.__dropout,
            'bidirectional': self.__bidirectional,
            'name': self.__name,
            'type': 'RNN',
            'layer': _RNN,
            "keyword_arguments":{
                    'input_size': self.__input_size,
                    'hidden_size': self.__hidden_size,
                    'num_layers': self.__num_layers
            }
        }
        
        

