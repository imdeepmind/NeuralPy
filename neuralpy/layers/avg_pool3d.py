"""AvgPool3D layer for NeuralPy"""

from torch.nn import AvgPool3d as _AvgPool3d


class AvgPool3D:
    """
        Applies a 2D average pooling over an input signal composed of several input planes.

        To learn more about Dense layers, please check PyTorch
        documentation https://pytorch.org/docs/stable/nn.html?highlight=AvgPool3D#torch.nn.AvgPool3d

        Supported Arguments:
            kernel_size: (Int | Tuple) Kernel size of the layer
                stride: (Int | Tuple) Controls the stride for the cross-correlation, a single
                        number or a one-element tuple.
                padding: (Int | Tuple) Controls the amount of implicit zero-paddings on both 
                            sides for padding number of points
                ceil_mode: (Bool) when True, will use ceil instead of floor to compute the output shape
                count_include_pad: (Bool) when True, will include the zero-padding in the averaging calculation
                divisor_override: (Bool) if specified, it will be used as divisor, otherwise attr:kernel_size will be used

    """

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None, name=None):
        """
            __init__ method for AvgPool3D

            Supported Arguments:
                kernel_size: (Int | Tuple) Kernel size of the layer
                stride: (Int | Tuple) Controls the stride for the cross-correlation, a single
                        number or a one-element tuple.
                padding: (Int | Tuple) Controls the amount of implicit zero-paddings on both 
                            sides for padding number of points
                ceil_mode: (Bool) when True, will use ceil instead of floor to compute the output shape
                count_include_pad: (Bool) when True, will include the zero-padding in the averaging calculation
                divisor_override: (Bool) if specified, it will be used as divisor, otherwise attr:kernel_size will be used

        """
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

            if isinstance(kernel_size[2], int):
                raise ValueError("Please provide a valid kernel_size")

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

            if isinstance(stride[2], int):
                raise ValueError("Please provide a valid stride")

        if stride is None:
            stride = kernel_size

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

            if isinstance(padding[2], int):
                raise ValueError("Please provide a valid padding")

        # Checking ceil_mode
        if not isinstance(ceil_mode, bool):
            raise ValueError("Please provide a valid ceil_mode")

        # Checking count_include_pad
        if not isinstance(count_include_pad, bool):
            raise ValueError("Please provide a valid count_include_pad")

        # Checking divisor_override
        if divisor_override is not None and not isinstance(divisor_override, int):
            raise ValueError("Please provide a valid divisor_override")

        # Checking the name field, this is an optional field,
        # if not provided generates a unique name for the layer
        if name is not None and not (isinstance(name, str) and name):
            raise ValueError("Please provide a valid name")

        # Storing the data
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__padding = padding
        self.__ceil_mode = ceil_mode
        self.__count_include_pad = count_include_pad
        self.__divisor_override = divisor_override

        self.__name = name

    def __get_layer_details(self):
        depth, width, height, fourth = self.__prev_layer_data

        # Getting the kernel_size
        k1 = k2 = k3 = 0
        if isinstance(self.__kernel_size, int):
            k1 = k2 = k3 = self.__kernel_size
        else:
            k1, k2, k3 = self.__kernel_size

        # Getting the padding values
        p1 = p2 = p3 = 0
        if isinstance(self.__padding, int):
            p1 = p2 = p3 = self.__padding
        else:
            p1, p2, p3 = self.__padding

        # Getting the stride values
        s1 = s2 = s3 = 0
        if isinstance(self.__stride, int):
            s1 = s2 = s3 = self.__stride
        else:
            s1, s2, s3 = self.__stride

        w1 = ((width + 2 * p1 - k1) // s1) + 1
        w2 = ((height + 2 * p2 - k2) // s2) + 1
        w3 = ((fourth + 2 * p3 - k3) // s3) + 1

        return (depth, depth * w1 * w2 * w3, (depth, w1, w2, w3))

    def get_input_dim(self, prev_input_dim, prev_layer_type):
        """
            This method calculates the input shape for layer based on previous output layer.

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # AvgPool3D does not need to n_input, so returning None
        layer_type = prev_layer_type.lower()

        if layer_type == 'conv3d':
            self.__prev_layer_data = prev_input_dim[2]

    def get_layer(self):
        """
            This method returns the details as dict of the layer.

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # Returning all the details of the layer
        return{
            'layer_details': self.__get_layer_details(),
            'layer': _AvgPool3d,
            'name': self.__name,
            'type': 'AvgPool3D',
            'keyword_arguments': {
                'kernel_size': self.__kernel_size,
                'stride': self.__stride,
                'padding': self.__padding,
                'ceil_mode': self.__ceil_mode,
                'count_include_pad': self.__count_include_pad,
                'divisor_override': self.__divisor_override
            }
        }
