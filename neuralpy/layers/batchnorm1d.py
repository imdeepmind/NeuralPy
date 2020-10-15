"""BatchNorm1d layer for NeuralPy"""

from torch.nn import BatchNorm1d as _BatchNorm1d


class BatchNorm1D:
    """
        Applies Batch Normalization over a 2D or 3D input
        To learn more about BatchNorm1d layers, please check PyTorch
        documentation at https://pytorch.org/docs/stable/nn.html#batchnorm1d

        Supported Arguments:
            num_features: (Integer) C from an expected input of size (N,C,L) or
                L from input of size (N,L)
            eps: (Float) a value added to the denominator for numerical stability.Default: 1e-5
            momentum: (Float) the value used for the running_mean and running_var computation.
                Can be set to None for cumulative moving average(i.e. simple average).Default: 0.1
            affine: (Boolean) a boolean value that when set to True,
                this module has learnable affine parameters. Default: True
            track_running_status: (Boolean) a boolean value that when set to True,
                this module tracks the running mean and variance, and when set to False,
                this module does not track such statistics and always uses batch statistics
                in both training and eval modes. Default: True
            name: (String) Name of the layer, if not provided then
                automatically calculates a unique name for the layer
    """
    def __init__(
            self, num_features, eps=1e-05, momentum=0.1, affine=True,
            track_running_status=True, name=None):
        """
            __init__ method for BatchNorm1d

            Supported Arguments:
                num_features: (Integer) C from an expected input of size (N,C,L) or
                    L from input of size (N,L)
                eps: (Float) A value added to the denominator for numerical stability.Default: 1e-5
                momentum: (Float) The value used for the running_mean and running_var computation.
                    Can be set to None for cumulative moving average(i.e. simple average).
                    Default: 0.1
                affine: (Boolean) A boolean value that when set to True,
                    this module has learnable affine parameters. Default: True
                track_running_status: (Boolean) A boolean value that when set to True,
                    this module tracks the running mean and variance, and when set to False,
                    this module does not track such statistics and always uses batch statistics
                    in both training and eval modes. Default: True
                name: (String) Name of the layer, if not provided then
                    automatically calculates a unique name for the layer
        """
        # Checking num_features field
        if num_features is not None and not isinstance(num_features, int):
            raise ValueError("Please provide a valid num_features")
        # Checking eps field
        if  not isinstance(eps, float):
            raise ValueError("Please provide a valid eps")
        # Checking momentum field
        if not isinstance(momentum, float):
            raise ValueError("Please provide a valid momentum")
        # Checking affine field
        if not isinstance(affine, bool):
            raise ValueError("Please provide a valid affine")
        # Checking test_running_status field
        if not isinstance(track_running_status, bool):
            raise ValueError("Please provide a vlaid track_running_status")
        # Checking name field
        if name is not None and not (isinstance(name, str) and name):
            raise ValueError("Please provide a valid name")

        # Storing Values
        self.__num_features = num_features

        self.__eps = eps
        self.__momentum = momentum
        self.__affine = affine
        self.__track_running_status = track_running_status
        self.__name = name

    def get_input_dim(self, prev_input_dim, prev_layer_type):
        """
            This method calculates the input shape for layer based on previous output layer.

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # based on the prev layer type, predicting the __num_features
        # Checking if num_features is there or not, not overwriting the num_features field
        if not self.__num_features:

            # based on the prev layer type, predicting the __num_features
            # to support more layers, we need to add some more statements
            layer_type = prev_layer_type.lower()
            if layer_type in ("dense", "conv1d", "conv2d"):
                self.__num_features = prev_input_dim[1]
            else:
                raise ValueError(
                    "Unsupported previos layer, please provide your own input shape for the layer")

        return self.__num_features

    def get_layer(self):
        """
            This method returns the details as dict of the layer.

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # Returning all the details of the layer
        return{
            "layer_details": (self.__num_features,),
            "name": self.__name,
            "layer": _BatchNorm1d,
            "type": "BatchNorm1d",
            "keyword_arguments":{
                "num_features": self.__num_features,
                "eps": self.__eps,
                "momentum": self.__momentum,
                "affine": self.__affine,
                "track_running_status": self.__track_running_status
            }

        }
