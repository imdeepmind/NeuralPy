"""Dropout Layer"""

from torch.nn import Dropout as _Dropout

class Dropout:
    """
        Applies Dropout to the input.

        The Dropout layer randomly sets input units to 0 with a frequency of `rate`
        at each step during training time, which helps prevent overfitting.
        Inputs not set to 0 are scaled up by 1/(1 - rate)
        such that the sum over all inputs is unchanged.

        Supported Arguments
            p=0.5: (Float) Probability of an element to be zeroed.
                    The value should be between 0.0 and 1.0.
            name=None: (String) Name of the layer, if not provided
                        then automatically calculates a unique name for the layer
    """
    def __init__(self, p=0.5, name=None):
        """
            __init__ method for Dropout class

            Supported Arguments
                p=0.5: (Float) Probability of an element to be zeroed.
                        The value should be between 0.0 and 1.0.
                name=None: (String) Name of the layer, if not provided
                            then automatically calculates a unique name for the layer
        """
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
        """
            This method calculates the input shape for layer based on previous output layer.

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # Dropout does not need to n_input, so returning None
        return None

    def get_layer(self):
        """
            This method returns the details as dict of the layer.

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
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
