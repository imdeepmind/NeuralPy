"""Sequential Models"""

from collections import OrderedDict
import torch

from .model import Model
from .model_helper import (is_valid_layer, build_layer_from_ref_and_details)

class Sequential(Model):
    """
        Sequential is a linear stack of layers with single
        input and output layer. It is one of the simplest types of models.
        In Sequential models, each layer has a single input and output tensor.

        Supported Arguments:
            force_cpu=False: (Boolean) If True, then uses CPU even if CUDA is available
            training_device=None: (NeuralPy device class) Device that will
                be used for training predictions
            random_state: (Integer) Random state for the device
    """

    def __init__(self, force_cpu=False, training_device=None, random_state=None):
        """
            __init__ method for Sequential Model

            Supported Arguments:
                force_cpu=False: (Boolean) If True, then uses CPU even if CUDA is available
                training_device=None: (NeuralPy device class) Device that will
                    be used for training predictions
                random_state: (Integer) Random state for the device
        """
        super(Sequential, self).__init__(force_cpu, training_device, random_state)
        # Initializing some attributes that we need to function
        self.__layers = []
        self.__build = False

    def add(self, layer):
        """
            In a Sequential model, the .add() method is responsible
            for adding a new layer to the model. It accepts a NeuralPy
            layer class as an argument and builds a model, and based on that.
            The .add() method can be called as many times as needed. There is no
            limitation on that, assuming you have enough computation power to handle it.

            Supported Arguments
                layer: (NeuralPy layer classes) Adds a layer into the model
        """
        # If we already built the model, then we can not a new layer
        if self.__build:
            raise Exception(
                "You have built this model already, you can not make any changes in this model")

        # Layer verification using the method is_valid_layer
        if not is_valid_layer(layer):
            raise ValueError("Please provide a valid neuralpy layer")

        # Finally adding the layer for layers array
        self.__layers.append(layer)

    def build(self):
        """
            In a Sequential model, the .build() method is responsible for
            building the PyTorch model from the NeuralPy model.

            After finishing the architecture of the model, the model
            needed to be build before training.

            Supported Arguments:
                There is no argument for this model
        """
        # Building the layers from the layer refs and details
        layers = build_layer_from_ref_and_details(self.__layers)

        # Making the PyTorch model using nn.Sequential
        model = torch.nn.Sequential(OrderedDict(layers))

        self.set_model(model)

        # Changing the build status to True, so we can not make any changes
        self.__build = True
