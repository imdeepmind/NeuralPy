import pytest
from torch.nn import CrossEntropyLoss as _CrossEntropyLoss
from neuralpy.loss_functions import CrossEntropyLoss

import numpy as np
import torch

# Possible values that are invalid
weights=["asd", 12, -.3]
reductions=["asdas", "", 12, 6.3]
ignore_indexes=["asd", False, "", 2.36]

@pytest.mark.parametrize(
	"weight, reduction, ignore_index", 
	[(weight, reduction, ignore_index) for weight in weights
								     for reduction in reductions
								     for ignore_index in ignore_indexes]
)
def test_cce_should_throw_value_error(weight, reduction, ignore_index):
    with pytest.raises(ValueError) as ex:
        x = CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)

# Possible values that are valid
weights=[[1.0, 1.0, 1.0], [2.0, 1.0, 2.0], np.ones([3])]
reductions=["mean"]
ignore_indexes=[1, -100, 5]

@pytest.mark.parametrize(
	"weight, reduction, ignore_index", 
	[(weight, reduction, ignore_index) for weight in weights
								     for reduction in reductions
								     for ignore_index in ignore_indexes]
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