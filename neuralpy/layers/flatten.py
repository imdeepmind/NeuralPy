"""Flatten layer for NeuralPy"""

from torch.nn import Flatten as _Flatten


class Flatten:
    """
        Flattens a contiguous range of dims into a tensor

        To learn more about Dense layers, please check PyTorch
        documentation https://pytorch.org/docs/stable/nn.html?highlight=flatten#torch.nn.Flatten

        Supported Arguments:
            start_dim: (Integer) first dim to flatten (default = 1)
            end_dim: (Integer)  last dim to flatten (default = -1)
    """
    def __init__(self, start_dim=1, end_dim=-1):
        """
            __init__ method for Flatten

            Supported Arguments:
                start_dim: (Integer) first dim to flatten (default = 1)
                end_dim: (Integer)  last dim to flatten (default = -1)

        """
        # Checking start_dim
        if start_dim and not(isinstance(start_dim, int)):
            raise ValueError("Please provide a valid start_dim")

        # Checking end_dim
        if end_dim and not(isinstance(end_dim, int)):
            raise ValueError("Please provide a vlaid end_dim")

        # Storing the data
        self.__start_dim = start_dim
        self.__end_dim = end_dim

    def get_input_dim(self, prev_input_dim, prev_layer_type):
        """
            This method calculates the input shape for layer based on previous output layer.

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # Flatten does not need to n_input, so returning None
        return None

    def get_layer(self):
        """
            This method returns the details as dict of the layer.

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # Returning all the details of the layer
        return{
            'layer_details': (self.__start_dim, self.__end_dim),
            'layer': _Flatten,
            'type': 'Flatten',
            'keyword_arguments': {
                'start_dim': self.__start_dim,
                'end_dim': self.__end_dim
            }
        }
