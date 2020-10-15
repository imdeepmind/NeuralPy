"""MaxPool3d for NeuralPy"""

from torch.nn import MaxPool3d as _MaxPool3d


class MaxPool3D:
    """
        MaxPool3d Applies a 3D max pooling over an input
        To learn more about MaxPool3d, please check pytorch
        documentation at https://pytorch.org/docs/stable/nn.html#maxpool3d

        Supported Arguments:

            kernel_size: (Int | Tuple) the size of the window to take a max over
            stride: (Int | Tuple) the stride of the window.
                Default value is kernel_size
            padding: (Int | Tuple) implicit zero padding to be
                added on both sides
            dilation: (Integer) a parameter that controls the
                stride of elements in the window
            return_indices: (Boolean) if True, will return the
                max indices along with the outputs
            ceil_mode: (Boolean) when True, will use ceil instead
                of floor to compute the output shape
            name: (String) Name of the layer, if not provided then
                automatically calculates a unique name for the layer
    """

    # pylint: disable=too-many-branches,too-many-statements
    def __init__(
            self, kernel_size, stride=None, padding=0, dilation=1,
            return_indices=False, ceil_mode=False, name=None
    ):
        """
            __init__ method for MaxPool3d

            Supported Arguments:

            kernel_size: (Integer) the size of the window to take a max over
            stride: (Integer) the stride of the window.
                Default value is kernel_size
            padding: (Integer) implicit zero padding to be
                added on both sides
            dilation: (Integer) a parameter that controls the
                stride of elements in the window
            return_indices: (Boolean) if True, will return the
                max indices along with the outputs
            ceil_mode: (Boolean) when True, will use ceil instead
                of floor to compute the output shape
        """
        # Checking the kernel_size field
        if not isinstance(kernel_size, (int, tuple)):
            raise ValueError("Please provide a valid kernel_size")

        if isinstance(kernel_size, tuple):
            if not isinstance(kernel_size[0], int):
                raise ValueError("Please provide a valid kernel_size")

            if not isinstance(kernel_size[1], int):
                raise ValueError("Please provide a valid kernel_size")

            if not isinstance(kernel_size[2], int):
                raise ValueError("Please provide a valid kernel_size")

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

        if stride is None:
            stride = kernel_size

        # Checking the padding field
        if not isinstance(padding, (int, tuple)):
            raise ValueError("Please provide a valid padding")

        if isinstance(padding, tuple):
            if not isinstance(padding[0], int):
                raise ValueError("Please provide a valid padding")

            if not isinstance(padding[1], int):
                raise ValueError("Please provide a valid padding")

            if not isinstance(padding[2], int):
                raise ValueError("Please provide a valid padding")

        # Checking the dilation field
        if not isinstance(dilation, (int, tuple)):
            raise ValueError("Please provide a valid dilation")

        if isinstance(dilation, tuple):
            if not isinstance(dilation[0], int):
                raise ValueError("Please provide a valid dilation")

            if not isinstance(dilation[1], int):
                raise ValueError("Please provide a valid dilation")

            if not isinstance(dilation[2], int):
                raise ValueError("Please provide a valid dilation")

        # Checking the return_indices, it is an optional filed
        if return_indices and not isinstance(return_indices, bool):
            raise ValueError(
                "Please provide a valid value for return_indices")

        # Checking the ceil_mode, it is an optional filed
        if ceil_mode and not isinstance(ceil_mode, bool):
            raise ValueError(
                "Please provide a valid value for ceil_mode")

        # Checking the name, it is an optional filed
        if name is not None and not (isinstance(name, str) and name):
            raise ValueError("Please provide a valid name")

        # Storing the data
        self.__kernel_size = kernel_size
        self.__stride = stride

        self.__padding = padding
        self.__dilation = dilation
        self.__return_indices = return_indices
        self.__ceil_mode = ceil_mode
        self.__name = name

        self.__prev_layer_data = None

    def __get_layer_details(self):
        depth, width, height, fourth = self.__prev_layer_data

        # Getting the kernel_size
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

        dim_1 = ((width + 2 * padding_1 - kernel_1) // stride_1) + 1
        dim_2 = ((height + 2 * padding_2 - kernel_2) // stride_2) + 1
        dim_3 = ((fourth + 2 * padding_3 - kernel_3) // stride_3) + 1

        return (depth, depth * dim_1 * dim_2 * dim_3, (depth, dim_1, dim_2, dim_3))

    def get_input_dim(self, prev_input_dim, prev_layer_type):
        """
            This method calculates the input shape for layer based on previous output layer.

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # MaxPool3d does not need to n_input, so returning None
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
            'name': self.__name,
            'type': 'MaxPool3D',
            'layer': _MaxPool3d,
            'keyword_arguments': {
                'kernel_size': self.__kernel_size,
                'stride': self.__stride,
                'padding': self.__padding,
                'dilation': self.__dilation,
                'return_indices': self.__return_indices,
                'ceil_mode': self.__ceil_mode
            }
        }
