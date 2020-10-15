"""Dense layer for NeuralPy"""

from torch.nn import Conv3d as _Conv3d


class Conv3D:
    """
        Applies a 3D convolution over an input signal composed of several input planes.

        To learn more about Conv3D layers, please check PyTorch
        documentation at https://pytorch.org/docs/stable/nn.html#Conv3d

        Supported Arguments:
            filters: (Integer) Size of the filter
            kernel_size: (Int | Tuple) Kernel size of the layer
            input_shape: (Tuple) A tuple with the shape in following format (input_channel, X, Y, Z)
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
    # pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-branches,too-many-statements
    def __init__(self, filters, kernel_size, input_shape=None,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, name=None):
        """
            __init__ method for the Conv3D layer

            Supported Arguments:
                filters: (Integer) Size of the filter
                kernel_size: (Int | Tuple) Kernel size of the layer
                input_shape: (Tuple) A tuple with the shape in following format
                    (input_channel, X, Y, Z) no need for this argument layers
                    except the initial layer.
                stride: (Int | Tuple) Controls the stride for the cross-correlation, a single
                        number or a one-element tuple.
                padding: (Int | Tuple) Controls the amount of implicit zero-paddings on both
                            sides for padding number of points
                dilation: (Int | Tuple) Controls the spacing between the kernel points; also
                            known as the à trous algorithm. It is harder to describe,
                            but this link has a nice visualization of what dilation does.
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
        if kernel_size is not None and not isinstance(kernel_size, (int, tuple)):
            raise ValueError("Please provide a valid kernel_size")

        if isinstance(kernel_size, tuple):
            if not isinstance(kernel_size[0], int):
                raise ValueError("Please provide a valid kernel_size")

            if not isinstance(kernel_size[1], int):
                raise ValueError("Please provide a valid kernel_size")

            if not isinstance(kernel_size[2], int):
                raise ValueError("Please provide a valid kernel_size")

        # Checking the input_shape field, it is a optional field
        if input_shape is not None and not isinstance(input_shape, tuple):
            raise ValueError("Please provide a valid input_shape")

        if input_shape is not None and not (isinstance(input_shape[0], int)
                                            and input_shape[0] >= 0):
            raise ValueError("Please provide a valid input_shape")

        if input_shape is not None and not (isinstance(input_shape[1], int)
                                            and input_shape[1] >= 0):
            raise ValueError("Please provide a valid input_shape")

        if input_shape is not None and not (isinstance(input_shape[2], int)
                                            and input_shape[2] >= 0):
            raise ValueError("Please provide a valid input_shape")

        if input_shape is not None and not (isinstance(input_shape[3], int)
                                            and input_shape[3] >= 0):
            raise ValueError("Please provide a valid input_shape")

        # Checking the stride field
        if stride is not None and not isinstance(stride, (int, tuple)):
            raise ValueError("Please provide a valid stride")

        if isinstance(stride, tuple):
            if not isinstance(stride[0], int):
                raise ValueError("Please provide a valid stride")

            if not isinstance(stride[1], int):
                raise ValueError("Please provide a valid stride")

            if not isinstance(stride[2], int):
                raise ValueError("Please provide a valid stride")

        # Checking the padding field
        if padding is not None and not isinstance(padding, (int, tuple)):
            raise ValueError("Please provide a valid padding")

        if isinstance(padding, tuple):
            if not isinstance(padding[0], int):
                raise ValueError("Please provide a valid padding")

            if not isinstance(padding[1], int):
                raise ValueError("Please provide a valid padding")

            if not isinstance(padding[2], int):
                raise ValueError("Please provide a valid padding")

        # Checking the dilation field
        if dilation is not None and not isinstance(dilation, (int, tuple)):
            raise ValueError("Please provide a valid dilation")

        if isinstance(dilation, tuple):
            if not isinstance(dilation[0], int):
                raise ValueError("Please provide a valid dilation")

            if not isinstance(dilation[1], int):
                raise ValueError("Please provide a valid dilation")

            if not isinstance(dilation[2], int):
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
        kernel_1 = kernel_2 = kernel_3 = 0
        if isinstance(self.__kernel_size, int):
            kernel_1 = kernel_2 = kernel_3 = self.__kernel_size
        else:
            kernel_1, kernel_2, kernel_3 = self.__kernel_size

        # Getting the padding values
        padding_1 = padding_2 = padding_3 = 0
        if isinstance(self.__padding, int):
            padding_1 = padding_2 = padding_3 = self.__padding
        else:
            padding_1, padding_2, padding_3 = self.__padding

        # Getting the stride values
        stride_1 = stride_2 = stride_3 = 0
        if isinstance(self.__stride, int):
            stride_1 = stride_2 = stride_3 = self.__stride
        else:
            stride_1, stride_2, stride_3 = self.__stride

        # Getting the dilation  values
        dilation_1 = dilation_2 = dilation_3 = 0
        if isinstance(self.__dilation, int):
            dilation_1 = dilation_2 = dilation_3 = self.__dilation
        else:
            dilation_1, dilation_2, dilation_3 = self.__dilation

        # Calculating the width and height of the conv output
        dim_1 = ((self.__input_shape[1] + 2 * padding_1 -
                  dilation_1 * (kernel_1 - 1) - 1) // stride_1) + 1
        dim_2 = ((self.__input_shape[2] + 2 * padding_2 -
                  dilation_2 * (kernel_2 - 1) - 1) // stride_2) + 1
        dim_3 = ((self.__input_shape[3] + 2 * padding_3 -
                  dilation_3 * (kernel_3 - 1) - 1) // stride_3) + 1

        # Returning for the next layers
        return (self.__input_shape[0], dim_1*dim_2*dim_3*self.__filters,
                (self.__filters, dim_1, dim_2, dim_3))

    def get_input_dim(self, prev_input_dim, prev_layer_type):
        """
            This method calculates the input shape for layer based on previous output layer.

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # Checking if n_inputs is there or not, not overwriting the n_input field
        if not self.__input_shape:
            layer_type = prev_layer_type.lower()

            # based on the prev layer type, predicting the __input_shape
            # to support more layers, we need to add some more statements
            if layer_type in ("conv3d", "avgpool3d", "maxpool3d", "batchnorm2d"):
                self.__input_shape = prev_input_dim[2]
            else:
                raise ValueError(
                    "Unsupported previous layer, please provide your own input shape for the layer"
                )

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
            'type': 'Conv3D',
            'layer': _Conv3d,
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
