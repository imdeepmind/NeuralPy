"""
Activation Functions are simple functions that tell a neuron to fire or not,
the purpose is to introduce non-linearity to Neural Network layers.

NeuralPy supports various activation functions that are widely
used for building complex Neural Network models.
"""

from .relu import ReLU
from .sigmoid import Sigmoid
from .softmax import Softmax
from .tanh import Tanh
from .leaky_relu import LeakyReLU
