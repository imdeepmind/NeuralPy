"""CustomLayer class for NeuralPy"""

import inspect


class CustomLayer:
    """
    CustomLayer is class for building custom NeuralPy classes

    Supported Arguments:
        layer_class: (Class) PyTorch class layer
        layer_type: (String) Type of a class, should be a string
        layer_name: (String) Name for the layer, optional
    """

    def __init__(self, layer_class, layer_type, layer_name=None):
        """
        __init__ method for CustomLayer

        Supported Arguments:
              layer_class: (Class) PyTorch class layer
              layer_type: (String) Type of a class, should be a string
              layer_name: (String) Name for the layer, optional
        """
        # TODO: In future, we might need to add more checks to
        # confirm this is a PyTorch valid class
        if not inspect.isclass(layer_class):
            raise ValueError("Please provide a valid layer class")

        if not (isinstance(layer_type, str) and layer_type):
            raise ValueError("Please provide a valid layer type")

        if layer_name is not None and not (isinstance(layer_name, str) and layer_name):
            raise ValueError("Please provide a valid name")

        self.__layer_type = layer_type
        self.__layer = layer_class
        self.__layer_name = layer_name

    def _get_layer_details(self, layer_details=None, keyword_arguments=None):
        """
        Creates the layer details data for the layers and returns an dictionary

        Supported Arguments:
            layer_details: (Tuple) Layers details tuple for next layer
            keyword_arguments: (Dict) Dict for layer keyword arguments
        """
        if layer_details is not None and not isinstance(layer_details, tuple):
            raise ValueError("Please provide a valid layer details tuple")

        if keyword_arguments is not None and not isinstance(keyword_arguments, dict):
            raise ValueError("Please provide a valid keyword arguments tuple")

        return {
            "layer_details": layer_details,
            "name": self.__layer_name,
            "type": self.__layer_type,
            "layer": self.__layer,
            "keyword_arguments": keyword_arguments,
        }

    def _get_layer_class(self):
        """
        Returns the layer class
        """
        return self.__layer

    def _get_layer_type(self):
        """
        Returns the layer type
        """
        return self.__layer_type

    def _get_layer_name(self):
        """
        Returns the layer name
        """
        return self.__layer_name
