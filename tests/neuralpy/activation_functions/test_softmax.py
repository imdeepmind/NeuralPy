import pytest
from torch.nn import Softmax as _Softmax
from neuralpy.activation_functions import Softmax

# Possible values
dims = [None]
names = [False, 12, 3.6, -2]

@pytest.mark.parametrize(
	"dim, name", 
	[(dim, name) for dim in dims
				 for name in names]
)
def test_leaky_relu_should_throw_value_error_Exception(dim, name):
    with pytest.raises(ValueError) as ex:
        x = Softmax(dim=dim, name=name)

# Possible values
dims = [0.01]
names = ["test1", "test2"]

@pytest.mark.parametrize(
    "dim, name", 
    [(dim, name) for dim in dims
                 for name in names]
)
def test_leaky_relu_get_input_dim_and_get_layer_method(dim, name):
    x = Softmax(dim=dim, name=name)

    assert x.get_input_dim(12) == None

    details = x.get_layer()

    assert isinstance(details, dict) == True

    assert details["n_inputs"] == None

    assert details["n_nodes"] == None

    assert details["name"] == name

    assert issubclass(details["layer"], _Softmax) == True

    assert isinstance(details["keyword_arguments"], dict) == True

    assert details["keyword_arguments"]["dim"] == dim