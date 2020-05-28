from torch.nn import Sigmoid as _Sigmoid
from .utils import get_activation_details


class Sigmoid:
    def __init__(self, name=None):
        # Checking the name field,
        # this is an optional field,
        # if not provided generates a unique name for the activation function
        if name is not None:
            if isinstance(name, str):
                if len(name) <= 0:
                    raise ValueError("Please provide a valid name")
            else:
                raise ValueError("Please provide a valid name")

        self.__name = name

    # pylint: disable=no-self-use,unused-argument
    def get_input_dim(self, prev_input_dim):
        # Sigmoid does not need to n_input, so returning None
        return None

    def get_layer(self):
        # Returning all the details of the activation function
        return get_activation_details(None, None, self.__name, 'Sigmoid', _Sigmoid, None)
