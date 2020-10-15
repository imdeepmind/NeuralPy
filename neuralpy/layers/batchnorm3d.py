"""BatchNorm3d layer for NeuralPy"""

from torch.nn import BatchNorm3d as _BatchNorm3d


class BatchNorm3D:
    """
        Applies Batch Normalization over a 5D input
        To learn more about BatchNorm1d layers, please check PyTorch
        documentation at https://pytorch.org/docs/stable/nn.html#batchnorm3d

        Supported Arguments:
            num_features: (Integer) C from an expected input of size (N,C,D,H,W)
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
            self, num_features=None, eps=1e-05, momentum=0.1, affine=True,
            track_running_stats=True, name=None):
        """
            __init__ method for BatchNorm3d

            Supported Arguments:
            num_features: (Integer) C from an expected input of size (N,C,D,H,W)
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
        # Checking num_features field
        if num_features is not None and not isinstance(num_features, int):
            raise ValueError("Please provide a valid num_features")
        # Checking eps field
        if not isinstance(eps, float):
            raise ValueError("Please provide a valid eps")
        # Checking momentum field
        if not isinstance(momentum, float):
            raise ValueError("Please provide a valid momentum")
        # Checking affine field
        if not isinstance(affine, bool):
            raise ValueError("Please provide a valid affine")
        # Checking test_running_status field
        if not isinstance(track_running_stats, bool):
            raise ValueError("Please provide a valid track_running_status")
        # Checking name field
        if name is not None and not (isinstance(name, str) and name):
            raise ValueError("Please provide a valid name")

        # Storing Values
        self.__num_features = num_features

        self.__eps = eps
        self.__momentum = momentum
        self.__affine = affine
        self.__track_running_stats = track_running_stats
        self.__name = name

    # pylint: disable=W0612
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
            if layer_type == 'conv3d':
                self.__num_features = prev_input_dim[2][0]
            else:
                raise ValueError(
                    "Unsupported previous layer, please provide your own input shape for the layer")

        return self.__num_features

    def get_layer(self):
        """
            This method returns the details as dict of the layer.

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # Returning all the details of the layer
        return{
            "layer_details": None,
            "name": self.__name,
            "layer": _BatchNorm3d,
            "type": "BatchNorm3d",
            "keyword_arguments": {
                "num_features": self.__num_features,
                "eps": self.__eps,
                "momentum": self.__momentum,
                "affine": self.__affine,
                "track_running_stats": self.__track_running_stats
            }
        }
