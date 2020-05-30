"""
    Optimizers are one of the most important parts of Machine Learning.
    The neuralpy.optimizer package implements different types of
    optimizers that can be used to optimizer a neuralpy model.

    Currently there 4 types of the optimizer, that neuralpy supports,
    these are Adam, Adagrad, RMSProp, and SGD.
"""

from .adam import Adam
from .sgd import SGD
from .adagrad import Adagrad
from .rmsprop import RMSprop
