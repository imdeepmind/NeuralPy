from torch.nn import Dropout as _Dropout

class Dropout:
    def __init__(self, p=0.5, name=None):
        # Checking the field p, it should be a valid prob value
        if not (isinstance(p, float) and p >= 0.0 and p <= 1.0):
            raise ValueError("Please provide a valid p value")
        # Checking the name field, this is an optional field,
        # if not provided generates a unique name for the layer
        if name is not None and not (isinstance(name, str) and name):
            raise ValueError("Please provide a valid name")

        self.__p = p
        self.__name = name

    # pylint: disable=no-self-use,unused-argument
    def get_input_dim(self, prev_input_dim):
        # Dropout does not need to n_input, so returning None
        return None

    def get_layer(self):
        # Returning all the details of the layer
        return {
            'n_inputs': None,
            'n_nodes': None,
            'name': self.__name,
            'type': 'Dropout',
            'layer': _Dropout,
            "keyword_arguments": {
                'p': self.__p,
                'inplace': False
            }
        }
