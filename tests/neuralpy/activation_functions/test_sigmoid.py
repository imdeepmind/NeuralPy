import pytest
from torch.nn import Sigmoid as _Sigmoid
from neuralpy.activation_functions import Sigmoid

# Possible values
names = [False, 12, 3.6, -2]

@pytest.mark.parametrize(
	"name", 
	[(name) for name in names]
)
def test_sigmoid_should_throw_value_error_exception(name):
    with pytest.raises(ValueError) as ex:
        x = Sigmoid(name=name)

# Possible values
names = ["test1", "test2"]

@pytest.mark.parametrize(
    "name", 
    [(name) for name in names]
)
def test_get_input_dim_and_get_layer_method(name):
    x = Sigmoid(name=name)

    assert x.get_input_dim(12, "dense") == None

    details = x.get_layer()

    assert isinstance(details, dict) == True

    assert details["layer_details"] == None

    assert details["name"] == name

    assert issubclass(details["layer"], _Sigmoid) == True

    assert details["keyword_arguments"] == None