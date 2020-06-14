import pytest
from torch.nn import BCEWithLogitsLoss as _BCEWithLogitsLoss
from neuralpy.loss_functions import BCELoss

import numpy as np
import torch

# Possible values that are invalid
weights=["asd", 12, -.3]
reductions=["asdas", "", 12, 6.3]
pos_weights=["asd", 12, -.3]

@pytest.mark.parametrize(
	"weight, reduction, pos_weight", 
	[(weight, reduction, pos_weight) for weight in weights
								     for reduction in reductions
								     for pos_weight in pos_weights]
)
def test_bce_should_throw_value_error(weight, reduction, pos_weight):
    with pytest.raises(ValueError) as ex:
        x = BCELoss(weight=weight, reduction=reduction, pos_weight=pos_weight)

# Possible values that are valid
weights=[[1.0, 1.0, 1.0], [2.0, 1.0, 2.0], np.ones([3])]
reductions=["mean"]
pos_weights=[[1.0, 1.0, 1.0], [2.0, 1.0, 2.0], np.ones([3])]

@pytest.mark.parametrize(
	"weight, reduction, pos_weight", 
	[(weight, reduction, pos_weight) for weight in weights
								     for reduction in reductions
								     for pos_weight in pos_weights]
)
def test_bce_get_layer_method(weight, reduction, pos_weight):
	x = BCELoss(weight=weight, reduction=reduction, pos_weight=pos_weight)
		
	details = x.get_loss_function()

	assert isinstance(details, dict) == True

	assert issubclass(details["loss_function"], _BCEWithLogitsLoss) == True

	assert isinstance(details["keyword_arguments"], dict) == True

	assert torch.all(torch.eq(details["keyword_arguments"]["weight"], torch.tensor(weight).float())) == True

	assert details["keyword_arguments"]["reduction"] == reduction

	assert torch.all(torch.eq(details["keyword_arguments"]["pos_weight"], torch.tensor(pos_weight).float())) == True

def test_bce_get_layer_method_with_default_parameters():
	x = BCELoss()
		
	details = x.get_loss_function()

	assert isinstance(details, dict) == True

	assert issubclass(details["loss_function"], _BCEWithLogitsLoss) == True

	assert isinstance(details["keyword_arguments"], dict) == True

	assert details["keyword_arguments"]["weight"] == None

	assert details["keyword_arguments"]["reduction"] == 'mean'

	assert details["keyword_arguments"]["pos_weight"] == None