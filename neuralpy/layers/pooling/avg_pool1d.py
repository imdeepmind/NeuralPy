"""AvgPool1D layer for NeuralPy"""

from torch.nn import AvgPool1d as _AvgPool1d
from neuralpy.utils import CustomLayer


class AvgPool1D(CustomLayer):
    """
    Applies a 1D average pooling over an input signal composed of several input
    planes.

    To learn more about Dense layers, please check PyTorch
    documentation
    https://pytorch.org/docs/stable/nn.html?highlight=AvgPool1D#torch.nn.AvgPool1d

    Supported Arguments:
        kernel_size: (Int | Tuple) Kernel size of the layer
            stride: (Int | Tuple) Controls the stride for the cross-correlation, a
                single number or a one-element tuple.
            padding: (Int | Tuple) Controls the amount of implicit zero-paddings on
                both sides for padding number of points
            ceil_mode: (Bool) when True, will use ceil instead of floor to compute
                        the output shape
            count_include_pad: (Bool) when True, will include the zero-padding
                            in the averaging calculation
            divisor_override: (Bool) if specified, it will be used as divisor,
                                otherwise attr:kernel_size will be used

    """

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        name=None,
    ):
        """
        __init__ method for AvgPool1D

        Supported Arguments:
            kernel_size: (Int | Tuple) Kernel size of the layer
            stride: (Int | Tuple) Controls the stride for the cross-correlation, a
                single number or a one-element tuple.
            padding: (Int | Tuple) Controls the amount of implicit zero-paddings on
                both sides for padding number of points
            ceil_mode: (Bool) when True, will use ceil instead of floor to compute
                the output shape
            count_include_pad: (Bool) when True, will include the zero-padding in
                the averaging calculation
            divisor_override: (Bool) if specified, it will be used as divisor,
                otherwise attr:kernel_size will be used
        """
        # Checking the kernel_size field
        if not isinstance(kernel_size, (int, tuple)):
            raise ValueError("Please provide a valid kernel_size")

        if isinstance(kernel_size, tuple):
            if not isinstance(kernel_size[0], int):
                raise ValueError("Please provide a valid kernel_size")

        # Checking the stride field
        if stride is not None and not isinstance(stride, (int, tuple)):
            raise ValueError("Please provide a valid stride")

        if isinstance(stride, tuple):
            if not isinstance(stride[0], int):
                raise ValueError("Please provide a valid stride")

        if stride is None:
            stride = kernel_size

        # Checking the padding field
        if not isinstance(padding, (int, tuple)):
            raise ValueError("Please provide a valid padding")

        if isinstance(padding, tuple):
            if not isinstance(padding[0], int):
                raise ValueError("Please provide a valid padding")

        # Checking ceil_mode
        if not isinstance(ceil_mode, bool):
            raise ValueError("Please provide a valid ceil_mode")

        # Checking count_include_pad
        if not isinstance(count_include_pad, bool):
            raise ValueError("Please provide a valid count_include_pad")

        super().__init__(_AvgPool1d, "AvgPool1d", layer_name=name)

        # Storing the data
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__padding = padding
        self.__ceil_mode = ceil_mode
        self.__count_include_pad = count_include_pad

        self.__prev_layer_data = None

    def __get_layer_details(self):
        depth, width = self.__prev_layer_data

        # Getting the kernel_size
        kernel_1 = 0
        if isinstance(self.__kernel_size, int):
            kernel_1 = self.__kernel_size
        else:
            (kernel_1,) = self.__kernel_size

        # Getting the padding values
        padding_1 = 0
        if isinstance(self.__padding, int):
            padding_1 = self.__padding
        else:
            (padding_1,) = self.__padding

        # Getting the stride values
        stride_1 = 0
        if isinstance(self.__stride, int):
            stride_1 = self.__stride
        else:
            (stride_1,) = self.__stride

        dim_1 = ((width + 2 * padding_1 - kernel_1) // stride_1) + 1

        return (depth, depth * dim_1, (depth, dim_1))

    def set_input_dim(self, prev_input_dim, prev_layer_type):
        """
        This method calculates the input shape for layer based on previous output
        layer.

        This method is used by the NeuralPy Models, for building the models.
        No need to call this method for using NeuralPy.
        """
        # AvgPool1D does not need to n_input, so returning None
        layer_type = prev_layer_type.lower()

        if layer_type == "conv1d":
            self.__prev_layer_data = prev_input_dim[2]

    def get_layer(self):
        """
        This method returns the details as dict of the layer.

        This method is used by the NeuralPy Models, for building the models.
        No need to call this method for using NeuralPy.
        """
        # Returning all the details of the layer
        return self._get_layer_details(
            self.__get_layer_details(),
            {
                "kernel_size": self.__kernel_size,
                "stride": self.__stride,
                "padding": self.__padding,
                "ceil_mode": self.__ceil_mode,
                "count_include_pad": self.__count_include_pad,
            },
        )
