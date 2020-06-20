import pytest
from torch.nn import BCEWithLogitsLoss as _BCEWithLogitsLoss
from neuralpy.loss_functions import BCELoss

import numpy as np
import torch

@pytest.mark.parametrize(
	"weight, reduction, pos_weight", 
	[("invalid", 12, -.3),
	(12, 12, -.3),
	([1.0, 1.0, 1.0], "sump", -.3),
	([1.0, 1.0, 1.0], 3, -.3),
	([1.0, 1.0, 1.0], "sum", -.3),
	([1.0, 1.0, 1.0], "sum", "invalid")]
)
def test_bce_should_throw_value_error(weight, reduction, pos_weight):
    with pytest.raises(ValueError) as ex:
        x = BCELoss(weight=weight, reduction=reduction, pos_weight=pos_weight)

@pytest.mark.parametrize(
	"weight, reduction, pos_weight", 
	[([1.0, 1.0, 1.0], "mean", np.ones([3])),
	(np.ones([3]), "mean", [2.0, 1.0, 2.0]),
	]
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