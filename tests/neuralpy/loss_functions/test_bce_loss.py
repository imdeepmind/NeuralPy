import pytest
from torch.nn import BCEWithLogitsLoss as _BCEWithLogitsLoss
from neuralpy.loss_functions import BCELoss

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
def test_adagrad_should_throw_value_error(weight, reduction, pos_weight):
    with pytest.raises(ValueError) as ex:
        x = BCELoss(weight=weight, reduction=reduction, pos_weight=pos_weight)
