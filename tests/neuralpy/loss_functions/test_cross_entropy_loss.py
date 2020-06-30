import pytest
from torch.nn import CrossEntropyLoss as _CrossEntropyLoss
from neuralpy.loss_functions import CrossEntropyLoss

import numpy as np
import torch

@pytest.mark.parametrize(
	"weight, reduction, ignore_index", 
	[
		("invalid", "invalid", "invalid"),
		(12, "invalid", "invalid"),
		(np.ones([3]), "invalid", "invalid"),
		(np.ones([3]), 12, "invalid"),
		(np.ones([3]), "mean", "mean"),
		(np.ones([3]), "sum", "sum"),
		(np.ones([3]), "none", "asd"),
		(np.ones([3]), "none", False)
	]
)
def test_cce_should_throw_value_error(weight, reduction, ignore_index):
    with pytest.raises(ValueError) as ex:
        x = CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)

@pytest.mark.parametrize(
	"weight, reduction, ignore_index", 
	[
		([2.0, 1.0, 2.0], "mean", -100),
		(np.ones([3]), "sum", 1),
		(np.ones([3]), "none", 1),
	]
)
def test_cce_get_layer_method(weight, reduction, ignore_index):
	x = CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
		
	details = x.get_loss_function()

	assert isinstance(details, dict) == True

	assert issubclass(details["loss_function"], _CrossEntropyLoss) == True

	assert isinstance(details["keyword_arguments"], dict) == True

	assert torch.all(torch.eq(details["keyword_arguments"]["weight"], torch.tensor(weight).float())) == True

	assert details["keyword_arguments"]["reduction"] == reduction

	assert details["keyword_arguments"]["ignore_index"] == ignore_index

def test_CrossEntropyLoss_get_layer_method_with_default_parameters():
	x = CrossEntropyLoss()
		
	details = x.get_loss_function()

	assert isinstance(details, dict) == True

	assert issubclass(details["loss_function"], _CrossEntropyLoss) == True

	assert isinstance(details["keyword_arguments"], dict) == True

	assert details["keyword_arguments"]["weight"] == None

	assert details["keyword_arguments"]["reduction"] == 'mean'

	assert details["keyword_arguments"]["ignore_index"] == -100