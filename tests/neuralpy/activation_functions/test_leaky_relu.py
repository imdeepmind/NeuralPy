import pytest
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

