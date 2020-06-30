"""Dense layer for NeuralPy"""

from torch.nn import Conv1d as _Conv1d


class Conv1D:
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

    def __init__(self, filters, kernel_size, input_shape=None, stride=1, padding=0, dilation=1, groups=1, bias=True, name=None):
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
        # Checking the filters field
        if not filters or not isinstance(filters, int) or filters <= 0:
            raise ValueError("Please provide a valid filters")

        # Checking the kernel_size field
        if not kernel_size or not (
            isinstance(kernel_size, int) or isinstance(kernel_size, tuple)
        ):
            raise ValueError("Please provide a valid kernel_size")

        # Checking the input_shape field, it is a optional field
        if input_shape is not None and not isinstance(input_shape, tuple):
            raise ValueError("Please provide a valid input_shape")
        
        if input_shape is not None and not (isinstance(input_shape[0], int) and input_shape[0] >= 0):
            raise ValueError("Please provide a valid input_shape")
        
        if input_shape is not None and not (isinstance(input_shape[1], int) and input_shape[1] >= 0):
            raise ValueError("Please provide a valid input_shape")
        
        if input_shape is not None and not (isinstance(input_shape[2], int) and input_shape[2] >= 0):
            raise ValueError("Please provide a valid input_shape")
        
        # Checking the stride field
        if stride is not None and not (
            isinstance(stride, int) or isinstance(stride, tuple)
        ):
            raise ValueError("Please provide a valid stride")

        # Checking the padding field
        if padding is not None and not (
            isinstance(padding, int) or isinstance(padding, tuple)
        ):
            raise ValueError("Please provide a valid padding")
        
        # Checking the dilation field
        if dilation is not None and not (
            isinstance(dilation, int) or isinstance(dilation, tuple)
        ):
            raise ValueError("Please provide a valid dilation")

        # Checking the groups field
        if groups is not None and not isinstance(groups, int) or groups <= 0:
            raise ValueError("Please provide a valid groups")

        # Checking the bias field, this is also optional, default to True
        if not isinstance(bias, bool):
            raise ValueError("Please provide a valid bias")

        # Checking the name field, this is an optional field,
        # if not provided generates a unique name for the layer
        if name is not None and not (isinstance(name, str) and name):
            raise ValueError("Please provide a valid name")

        # Storing the data
        self.__filters = filter
        self.__kernel_size = kernel_size
        self.__input_shape = input_shape
        self.__stride = stride
        self.__padding = padding
        self.__dilation = dilation
        self.__groups = groups,

        self.__bias = bias
        self.__name = name

    def get_input_dim(self, prev_input_dim, prev_layer_type):
        """
            This method calculates the input shape for layer based on previous output layer.

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # Checking if n_inputs is there or not, not overwriting the n_input field
        if not self.__n_inputs:
            layer_type = prev_layer_type.lower()

            # based on the prev layer type, predicting the n_inputs
            # to support more layers, we need to add some more statements
            if layer_type == "dense":
                self.__n_inputs = prev_input_dim[0]
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
        return {
            'layer_details': (self.__n_nodes,),
            'name': self.__name,
            'type': 'Dense',
            'layer': Linear,
            "keyword_arguments": {
                'in_features': self.__n_inputs,
                'out_features': self.__n_nodes,
                'bias': self.__bias
            }
        }