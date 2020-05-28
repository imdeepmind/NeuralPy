import pytest
from torch.nn import LeakyReLU as _LeakyReLU
from neuralpy.activation_functions import LeakyReLU

# Possible values
negative_slopes = [0.01]
names = [False, 12, 3.6, -2]

@pytest.mark.parametrize(
	"negative_slope, name", 
	[(negative_slope, name) for negative_slope in negative_slopes
						    for name in names]
)
def test_leaky_relu_should_throw_value_error_Exception(negative_slope, name):
    with pytest.raises(ValueError) as ex:
        x = LeakyReLU(negative_slope=negative_slope, name=name)

# Possible values
negative_slopes = [0.01]
names = ["test1", "test2"]

@pytest.mark.parametrize(
    "negative_slope, name", 
    [(negative_slope, name) for negative_slope in negative_slopes
                            for name in names]
)
def test_leaky_relu_get_input_dim_and_get_layer_method(negative_slope, name):
    x = LeakyReLU(negative_slope=negative_slope, name=name)

    assert x.get_input_dim(12) == None

    details = x.get_layer()

    assert isinstance(details, dict) == True

    assert details["n_inputs"] == None

    assert details["n_nodes"] == None

    assert details["name"] == name

    assert issubclass(details["layer"], _LeakyReLU) == True

    assert isinstance(details["keyword_arguments"], dict) == True

    assert details["keyword_arguments"]["negative_slope"] == negative_slope

    assert details["keyword_arguments"]["inplace"] == False