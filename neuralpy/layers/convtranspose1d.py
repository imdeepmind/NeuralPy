"""ConvTranspose1d Layer for NeuralPy"""

from torch.nn import ConvTranspose1d as _ConvTranspose1d


class ConvTranspose1d:
    """

    """

    def __init__(self, in_channels, out_channels, kernel_size,
                stride=1, padding=0, output_padding=0,
                groups=1, bias=True, dilation=1, name=None):
        

        if not in_channels or not isinstance(in_channels, int)
                or in_channels < 0:
            raise ValueError("Please provide a valid in_channels")

        if not out_channels or not isinstance(out_channels, int)
                or out_channels < 0:
            raise ValueError("Please provide a valid out_channels")

        if not kernel_size or not isinstance(kernel_size, (int, tuple)):
            raise ValueError("Please provide a valid kernel_size")


        if stride is not None and not isinstance(stride, (int, tuple)):
            raise ValueError("Please provide a valid stride")

        if padding is not None and not isinstance(padding, (int, tuple)):
            raise ValueError("Please provide a valid padding")

        if output_padding is not None and not isinstance(output_padding, int, tuple):
            raise ValueError("Please provide a valid output_padding")

        if groups is not None and not isinstance(groups, int):
            raise ValueError("Please provide a valid groups")

        if bias is not None and not isinstance(bias, bool):
            raise ValueError("Please provide a valid bias")

        if dilation is not None and not isinstance(dilation, isinstance(dilation, (int, tuple))):
            raise ValueError("Please provide a valid dilation")

        if name is not None and not (isinstance(name, str) and name):
            raise ValueError("Please provide a valid name")


        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__kernel_size = kernel_size

        self.__stride = stride
        self.__padding = padding
        self.__out_padding = output_padding
        self.__groups = groups
        self.__bias = bias
        self.__dilation = dilation
        self.__name = name

    def __get_layer_details(self):
        Lout​=(Lin​−1)×stride−2×padding+dilation×(kernel_size−1)+output_padding+1

        kernel1 = 0
        if isinstance(self.__kernel_size, int):
            kernel1 = self.__kernel_size
        else:
            kernel1, = self.__kernel_size

        padding1 = 0
        if isinstance(self.__padding, int):
            padding1 = self.__padding
        else:
            padding1, = self.__padding
        
        dilation1 = 0
        if isinstance(self.__dilation, int):
            dilation1  = self.__dilation
        else:
            dilation1, = self.__dilation
        
        out_padding1 = 0
        if isinstance(self.__out_padding, int):
            out_padding1 = self.__out_padding
        else:
            out_padding1, = self.__out_padding
        
        # dim1 = (self.__input)

    def get_input_dim(self, prev_input_dim, prev_layer_type):
        pass

    def get_layer(self):

        return{
                'layer_details': self.__get_layer_details(),
                'name': self.__name,
                'type': 'ConvTranspose1d',
                'layer': _ConvTranspose1d,
                'keyword_arguments':{
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
                }
        }

