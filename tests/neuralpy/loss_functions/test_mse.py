import pytest
from torch.nn import MSELoss as _MSELoss
from neuralpy.loss_functions import MSELoss

import numpy as np
import torch

# Possible values that are invalid
reductions=["asdas", "", 12, 6.3]

@pytest.mark.parametrize(
	"reduction", 
	[(reduction) for reduction in reductions]
)
def test_cce_should_throw_value_error(reduction):
    with pytest.raises(ValueError) as ex:
        x = MSELoss(reduction=reduction)

# Possible values that are valid
reductions=["mean"]

@pytest.mark.parametrize(
	"reduction", 
	[(reduction) for reduction in reductions]
)
def test_mse_get_layer_method(reduction):
	x = MSELoss(reduction=reduction)
		
	details = x.get_loss_function()

	assert isinstance(details, dict) == True

	assert issubclass(details["loss_function"], _MSELoss) == True

	assert isinstance(details["keyword_arguments"], dict) == True

	assert details["keyword_arguments"]["reduction"] == reduction

def test_mse_get_layer_method_without_parameters():
	x = MSELoss()

	details = x.get_loss_function()

	assert isinstance(details, dict) == True

	assert issubclass(details["loss_function"], _MSELoss) == True

	assert isinstance(details["keyword_arguments"], dict) == True

	assert details["keyword_arguments"]["reduction"] == 'mean'