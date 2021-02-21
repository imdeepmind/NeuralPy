"""ConvTranspose3d Layer for NeuralPy"""

from torch.nn import ConvTranspose3d as _ConvTranspose3d
from neuralpy.utils import CustomLayer


class ConvTranspose3d(CustomLayer):
    """
    Applies a 3D transposed convolution operator over
    an input image composed of several input planes.

    To learn more about ConvTranspose3d layers, please check PyTorch
    documentation at
    https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html?
    highlight=convtr#torch.nn.ConvTranspose3d

    Supported Arguments:
        in_channels: (Integer) Number of channels in the input image
        out_channels: (Integer)  Number of channels produced by the convolution
        kernel_size: (Integer | Tuple)  Size of the convolving kernel
        stride: (Integer | Tuple)  Stride of the convolution. Default: 1
        padding: (Integer | Tuple)  dilation * (kernel_size - 1) - padding
            zero-padding will be added to both sides of the input. Default: 0
        output_padding: (Integer | Tuple)  Additional size added to one side
            of the output shape. Default: 0
        groups: (Integer)  Number of blocked connections from input channels
            to output channels. Default: 1
        bias: (Boolean)  If True, adds a learnable bias to the output.
            Default: True
        dilation: (Integer | Tuple)  Spacing between kernel elements. Default: 1
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        input_shape=None,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        name=None,
    ):
        """
            __init__ method for ConvTranspose3d layer

            Supported Arguments:
                in_channels: (Integer) Number of channels in the input image
                out_channels: (Integer)  Number of channels produced
                    by the convolution
                kernel_size: (Integer | Tuple)  Size of the convolving kernel
                stride: (Integer | Tuple)  Stride of the convolution. Default: 1
                padding: (Integer | Tuple)  dilation * (kernel_size - 1) - padding
                    zero-padding will be added to both sides of the input. Default: 0
                output_padding: (Integer | Tuple)  Additional size added to one side
                    of the output shape. Default: 0
                groups: (Integer)  Number of blocked connections from input channels
                    to output channels. Default: 1
                bias: (Boolean)  If True, adds a learnable bias to the output.
                    Default: True
                dilation: (Integer | Tuple)  Spacing between kernel elements.
                    Default: 1
        """
        # Checking the in_channels field
        if not in_channels or not isinstance(
                in_channels, int) or in_channels < 0:
            raise ValueError("Please provide a valid in_channels")

        # Checking the out_channels field
        if not out_channels or not isinstance(
                out_channels, int) or out_channels < 0:
            raise ValueError("Please provide a valid out_channels")

        # Checking the kernel_size field
        if not kernel_size or not isinstance(kernel_size, (int, tuple)):
            raise ValueError("Please provide a valid kernel_size")

        # Checking the input_shape field, it is a optional field
        if input_shape is not None and not isinstance(input_shape, tuple):
            raise ValueError("Please provide a valid input_shape")

        if input_shape is not None and not (
            isinstance(input_shape[0], int) and input_shape[0] >= 0
        ):
            raise ValueError("Please provide a valid input_shape")

        if input_shape is not None and not (
            isinstance(input_shape[1], int) and input_shape[1] >= 0
        ):
            raise ValueError("Please provide a valid input_shape")

        # Checking the stride field, it is a optional field
        if stride is not None and not isinstance(stride, (int, tuple)):
            raise ValueError("Please provide a valid stride")

        # Checking the padding field, it is a optional field
        if padding is not None and not isinstance(padding, (int, tuple)):
            raise ValueError("Please provide a valid padding")

        # Checking the output_padding field, it is a optional field
        if output_padding is not None and not isinstance(
                output_padding, (int, tuple)):
            raise ValueError("Please provide a valid output_padding")

        # Checking the groups field, it is a optional field
        if groups is not None and not isinstance(groups, int):
            raise ValueError("Please provide a valid groups")

        # Checking the bias field, it is a optional field
        if bias is not None and not isinstance(bias, bool):
            raise ValueError("Please provide a valid bias")

        # Checking the dilation field, it is a optional field
        if dilation is not None and not isinstance(
                dilation, (int, tuple)):
            raise ValueError("Please provide a valid dilation")

        super().__init__(_ConvTranspose3d, "ConvTranspose3d", layer_name=name)

        # Storing the data
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__kernel_size = kernel_size

        self.__input_shape = input_shape
        self.__stride = stride
        self.__padding = padding
        self.__out_padding = output_padding
        self.__groups = groups
        self.__bias = bias
        self.__dilation = dilation

    def __get_layer_details(self):
        # Return tuple structure
        # Getting the kernel values
        kernel1 = 0
        kernel2 = 0
        if isinstance(self.__kernel_size, int):
            kernel1 = self.__kernel_size
        else:
            kernel1, kernel2 = self.__kernel_size[0], self.__kernel_size[1]

        # Getting the padding values
        padding1 = 0
        padding2 = 0
        if isinstance(self.__padding, int):
            padding1 = self.__padding
        else:
            padding1, padding2 = self.__padding[0], self.__padding[1]

        # Getting the stride values
        stride_1 = 0
        stride_2 = 0
        if isinstance(self.__stride, int):
            stride_1 = self.__stride
            stride_2 = self.__stride
        else:
            stride_1, stride_2 = self.__stride[0], self.__stride[1]

        # Getting the dilation values
        dilation1 = 0
        dilation2 = 0
        if isinstance(self.__dilation, int):
            dilation1 = self.__dilation
            dilation2 = self.__dilation
        else:
            dilation1, dilation2 = self.__dilation[0], self.__dilation[1]

        # Getting the out_padding values
        out_padding1 = 0
        out_padding2 = 0
        if isinstance(self.__out_padding, int):
            out_padding1 = self.__out_padding
            out_padding2 = self.__out_padding
        else:
            out_padding1, out_padding2 = self.__out_padding[0], self.__out_padding[1]

        # Calculating the width and height of the convtranspose output
        input_shape1 = self.__input_shape[-1] - 1
        input_shape2 = self.__input_shape[-2] - 1
        input_shape3 = self.__input_shape[-3] - 1
        stride_1 = stride_1 - 2
        stride_2 = stride_2 - 2
        kernel1 = kernel1 - 1
        kernel2 = kernel2 - 1
        out_padding1 = out_padding1 + 1
        out_padding2 = out_padding2 + 1
        dim1 = input_shape1 * stride_1 * padding1 \
            + dilation1 * kernel1 + out_padding1
        
        dim2 = input_shape2 * stride_2 * padding2 \
            + dilation2 * kernel2 + out_padding2
        
        dim3 = input_shape3 * stride_2 * padding2 \
            + dilation2 * kernel2 + out_padding2

        return(
            self.__input_shape[0], self.__in_channels * dim1 * dim2, dim3,
            (self.__in_channels, dim1, dim2, dim3)
        )

    def get_input_dim(self, prev_input_dim, prev_layer_type):
        """
        This method calculates the input shape for layer based on previous output
        layer.

        This method is used by the NeuralPy Models, for building the models.
        No need to call this method for using NeuralPy.
        """
        # Checking if n_inputs is there or not, not overwriting the n_input
        # field
        if not self.__input_shape:
            layer_type = prev_layer_type.lower()

            # based on the prev layer type, predicting the __input_shape
            # to support more layers, we need to add some more statements
            if layer_type == 'convtranspose3d':
                self.__input_shape = prev_input_dim[-1]
            else:
                raise ValueError(
                    "Unsupported previous layer, please provide your own input shape\
                        for the layer"
                    )

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
                'in_channels': self.__in_channels,
                'out_channels': self.__out_channels,
                'kernel_size': self.__kernel_size,
                'stride': self.__stride,
                'padding': self.__padding,
                'out_padding': self.__out_padding,
                'groups': self.__groups,
                'bias': self.__bias,
                'dilation': self.__dilation,
                'padding_mode': 'zeros'
            },
        )
