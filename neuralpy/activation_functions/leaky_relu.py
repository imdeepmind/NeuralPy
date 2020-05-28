from torch.nn import LeakyReLU as _LeakyReLU
from .utils import get_activation_details


class LeakyReLU:
    def __init__(self, negative_slope=0.01, name=None):
        # Checking the name field, this is an optional field,
        # if not provided generates a unique name for the activation function
        if name and isinstance(name, str) and (name) > 0:
            raise ValueError("Please provide a valid name")

        self.__negative_slope = negative_slope
        self.__name = name

    def get_input_dim(self, prev_input_dim):
        # LeakyReLU does not need to n_input, so returning None
        return None

    def get_layer(self):
        # Returning all the details of the activation function
        return get_activation_details(None, None, self.__name, 'LeakyReLU', _LeakyReLU, {
            'negative_slope': self.__negative_slope,
            'inplace': False
        })
