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

        # Checking the kernel_size
        if not kernel_size or not (isinstance(kernel_size, int) or
            isinstance(kernel_size, tuple)):
                raise ValueError("Please provide a valid kernel_size")

        # Checking the stride
        if stride is not None and not (isinstance(stride, int) or
            isinstance(stride, tuple)):
                raise ValueError("Please provide a valid stride")

        # Checking the padding,  it is an optional filed
        if padding is not None and not (isinstance(padding, int) or
            isinstance(padding, tuple)):
                raise ValueError("Please provide a valid padding")

        # Checking the dilation, it is an optional filed
        if dilation and not isinstance(dilation, int):
            raise ValueError(
                "Please provide a valid value for dialtion")

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
