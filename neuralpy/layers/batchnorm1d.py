"""BatchNorm layer for NeuralPy"""

from torch.nn import BatchNorm1d as _BatchNorm1d


class BatchNorm1d:

    def __init__(
        self, num_features, eps=1e-05, momentum=0.1, affine=True,
        track_running_status=True, name=None):

        if not num_features or not isinstance(num_features, int):
            raise ValueError("Please provide a valid num_features")

        if eps and not isinstance(eps, float):
            raise ValueError("Please provide a valid eps")

        if momentum and not isinstance(momentum, float):
            raise ValueError("Please provide a valid momentum")

        if affine and not isinstance(affine, bool):
            raise ValueError("Please provide a valid affine")

        if track_running_status and not isinstance(track_running_status, bool):
            raise ValueError("Please provide a vlaid track_running_status")

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

        if not self.__n_inputs:
            layer_type = prev_layer_type.lower()

            if layer_type in (
                    "dense", "bilinear", "rnn", "gru", "lstm", "conv1d"):
                self.__num_features = prev_input_dim[-1]
            elif layer_type == "conv2d":
                self.__num_features = prev_input_dim[1]
            else:
                raise ValueError(
                    "Unsupported previous layer, please provide your own input shape for the layer")
    
    def get_layer(self):

        return{
            "layer_details": self.__num_features,
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