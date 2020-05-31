"""
	Loss Functions are functions that calculate the error rate of a model.
	The optimizer optimizes the model based on these Loss Functions.

	NeuralPy currently supports 3 types of Loss Functions,
	BCELoss, CrossEntropyLoss, and MeanSquaredLoss.
"""

from .cross_entropy import CrossEntropyLoss
from .mse import MSELoss
from .bce_loss import BCELoss
