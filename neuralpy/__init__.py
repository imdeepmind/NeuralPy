"""
	NeuralPy is a Keras like, machine learning library that works
	on top of PyTorch written purely in Python. It is simple,
	easy to use library, cross-compatible with PyTorch models,
	suitable for all kinds of machine learning experiments,
	learning, research, etc.
"""

# pylint: disable=no-name-in-module
from torch import device
from . import activation_functions
from . import layers
from . import loss_functions
from . import models
from . import optimizer
from . import regulariziers
