"""Dense layer for NeuralPy"""

from torch.nn import Conv2d as _Conv2d


class Conv2D:
    """
        Applies a 2D convolution over an input signal composed of several input planes.

        To learn more about Conv2D layers, please check PyTorch
        documentation at https://pytorch.org/docs/stable/nn.html#Conv2d

        Supported Arguments:
            filters: (Integer) Size of the filter
            kernel_size: (Int | Tuple) Kernel size of the layer
            input_shape: (Tuple) A tuple with the shape in following format (input_channel, X, y)
                no need for this argument layers except the initial layer.
            stride: (Int | Tuple) Controls the stride for the cross-correlation, a single
                    number or a one-element tuple.
            padding: (Int | Tuple) Controls the amount of implicit zero-paddings on both 
                        sides for padding number of points
            dilation: (Int | Tuple) Controls the spacing between the kernel points; also 
                        known as the à trous algorithm. It is harder to describe, but this link has 
                        a nice visualization of what dilation does.
            groups: (Int) Controls the connections between inputs and outputs. 
                    input channel and filters must both be divisible by groups
            bias: (Boolean) If true then uses the bias, Defaults to `true`
            name: (String) Name of the layer, if not provided then
                automatically calculates a unique name for the layer
    """

    def __init__(self, filters, kernel_size, input_shape=None, stride=1, padding=0, dilation=1, groups=1, bias=True, name=None):
        """
            __init__ method for the Conv2D layer

            Supported Arguments:
                filters: (Integer) Size of the filter
                kernel_size: (Int | Tuple) Kernel size of the layer
                input_shape: (Tuple) A tuple with the shape in following format (input_channel, X, y)
                    no need for this argument layers except the initial layer.
                stride: (Int | Tuple) Controls the stride for the cross-correlation, a single
                        number or a one-element tuple.
                padding: (Int | Tuple) Controls the amount of implicit zero-paddings on both 
                            sides for padding number of points
                dilation: (Int | Tuple) Controls the spacing between the kernel points; also 
                            known as the à trous algorithm. It is harder to describe, but this link has 
                            a nice visualization of what dilation does.
                groups: (Int) Controls the connections between inputs and outputs. 
                        input channel and filters must both be divisible by groups
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

    def __get_layer_details(self):
        # Return tuple structure
        return (self.__input_shape[0], self.__input_shape[1],
                self.__input_shape[2], self.__kernel_size, self.__stride, self.__padding)
        k = 0
        if isinstance(self.__kernel_size, int):
            k = self.__kernel_size
        else:
            k = self.__kernel_size[0]

        w = (self.__input_shape[1] - k +
             (2 * self.__padding) / self.__stride) + 1

        return (self.__input_shape[0], w*w*self.__filters, (self.__input_shape[0], w, w))

    def get_input_dim(self, prev_input_dim, prev_layer_type):
        """
            This method calculates the input shape for layer based on previous output layer.

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # Checking if n_inputs is there or not, not overwriting the __input_shape field
        if not self.__input_shape:
            layer_type = prev_layer_type.lower()

            print("From get_input_dim", prev_input_dim, prev_layer_type)

            # based on the prev layer type, predicting the __input_shape
            # to support more layers, we need to add some more statements
            if layer_type == "Conv2D":
                self.__input_shape = prev_input_dim[2]
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
            'layer_details': self.__get_layer_details(),
            'name': self.__name,
            'type': 'Conv2D',
            'layer': _Conv2d,
            "keyword_arguments": {
                'in_channels': self.__input_shape[0],
                'out_channels': self.__filters,
                'kernel_size': self.__kernel_size,
                'stride': self.__stride,
                'padding': self.__padding,
                'dilation': self.__dilation,
                'groups': self.__groups,
                'bias': self.__bias,
                'padding_mode': 'zeros'
            }
        }
