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

        if isinstance(kernel_size, tuple):
            if isinstance(kernel_size[0], int):
                raise ValueError("Please provide a valid kernel_size")

            if isinstance(kernel_size[1], int):
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

        if isinstance(stride, tuple):
            if isinstance(stride[0], int):
                raise ValueError("Please provide a valid stride")

            if isinstance(stride[1], int):
                raise ValueError("Please provide a valid stride")

        # Checking the padding field
        if padding is not None and not (
            isinstance(padding, int) or isinstance(padding, tuple)
        ):
            raise ValueError("Please provide a valid padding")

        if isinstance(padding, tuple):
            if isinstance(padding[0], int):
                raise ValueError("Please provide a valid padding")

            if isinstance(padding[1], int):
                raise ValueError("Please provide a valid padding")

        # Checking the dilation field
        if dilation is not None and not (
            isinstance(dilation, int) or isinstance(dilation, tuple)
        ):
            raise ValueError("Please provide a valid dilation")

        if isinstance(dilation, tuple):
            if isinstance(dilation[0], int):
                raise ValueError("Please provide a valid dilation")

            if isinstance(dilation[1], int):
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
        self.__filters = filters
        self.__kernel_size = kernel_size
        self.__input_shape = input_shape
        self.__stride = stride
        self.__padding = padding
        self.__dilation = dilation
        self.__groups = groups

        self.__bias = bias
        self.__name = name

    def __get_layer_details(self):
        # Return tuple structure
        # Getting the kernel values
        k1 = k2 = 0
        if isinstance(self.__kernel_size, int):
            k1 = k2 = self.__kernel_size
        else:
            k1, k2 = self.__kernel_size

        # Getting the padding values
        p1 = p2 = 0
        if isinstance(self.__padding, int):
            p1 = p2 = self.__padding
        else:
            p1, p2 = self.__padding

        # Getting the stride values
        s1 = s2 = 0
        if isinstance(self.__stride, int):
            s1 = s2 = self.__stride
        else:
            s1, s2 = self.__stride

        # Getting the dilation  values
        d1 = d2 = 0
        if isinstance(self.__dilation, int):
            d1 = d2 = self.__dilation
        else:
            d1, d2 = self.__dilation

        # Calculating the width and height of the conv output
        w1 = ((self.__input_shape[1] + 2 * p1 - d1 * (k1 - 1) - 1) // s1) + 1
        w2 = ((self.__input_shape[2] + 2 * p2 - d2 * (k2 - 1) - 1) // s2) + 1

        # Returning for the next layers
        return (self.__input_shape[0], w1*w2*self.__filters, (self.__filters, w1, w2))

    def get_input_dim(self, prev_input_dim, prev_layer_type):
        """
            This method calculates the input shape for layer based on previous output layer.

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # Checking if n_inputs is there or not, not overwriting the __input_shape field
        if not self.__input_shape:
            layer_type = prev_layer_type.lower()

            # based on the prev layer type, predicting the __input_shape
            # to support more layers, we need to add some more statements
            if layer_type == "conv2d" or layer_type == 'avgpool2d':
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
