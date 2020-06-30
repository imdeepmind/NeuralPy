import pytest
from torch.nn import Linear
from neuralpy.layers import Dense

def test_dense_should_throw_type_error():
    with pytest.raises(TypeError) as ex:
        x = Dense()

@pytest.mark.parametrize(
	"n_nodes, n_inputs, bias, name", 
	[
		(0.3, 0.3, 0.36, False),
		(False, 0.3, 0.36, False),
		(4, 0.3, 0.36, False),
		(10, False, 0.36, False),
		(10, "invalid", 0.36, False),
		(10, 10, 0.36, False),
		(10, 10, "invalid", False),
		(10, 10, True, False),
		(10, 10, True, 19),
		(10, 10, True, ""),
	]
)
def test_dense_should_throw_value_error(n_nodes, n_inputs, bias, name):
    with pytest.raises(ValueError) as ex:
        x = Dense(n_nodes=n_nodes, n_inputs=n_inputs, bias=bias, name=name)


# Possible values
n_nodes = [6, 3]
n_inputs = [6, 5, None]
biases = [True, False]
names = ["Test", None]

@pytest.mark.parametrize(
	"n_nodes, n_inputs, bias, name", 
	[(n_node, n_input, bias, name) for n_node in n_nodes
						           for n_input in n_inputs
						           for bias in biases
						           for name in names]
)
def test_dense_get_layer_method(n_nodes, n_inputs, bias, name):
	x = Dense(n_nodes=n_nodes, n_inputs=n_inputs, bias=bias, name=name)
	prev_dim = (6,)

	if n_inputs is None:
		x.get_input_dim(prev_dim, "dense")
		
	details = x.get_layer()

	assert isinstance(details, dict) == True

	assert details["layer_details"] == (n_nodes,)

	assert details["name"] == name

	assert issubclass(details["layer"], Linear) == True

	assert isinstance(details["keyword_arguments"], dict) == True

	if n_inputs:
		assert details["keyword_arguments"]["in_features"] == n_inputs
	else:
		assert details["keyword_arguments"]["in_features"] == prev_dim[0]

	assert details["keyword_arguments"]["out_features"] == n_nodes

	assert details["keyword_arguments"]["bias"] == bias