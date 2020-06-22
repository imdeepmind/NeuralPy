import pytest
from torch.nn import Bilinear as BiLinear
from neuralpy.layers import Bilinear

def test_bilinear_should_throw_type_error():
    with pytest.raises(TypeError) as ex:
        x = Bilinear()

n_nodes = [0.3, 6.3, -0.36, 'asd', '', False]
n_inputs = [0.3, 6.3, -0.36, 'asd', '', False]
n_inputs2 = [0.3, 6.3, -0.36, 'asd', '', False]
biases = [1, "", 0.3]
names = [False, 12, ""]

@pytest.mark.parametrize(
    "n_nodes, n_inputs, n_inputs2, bias, name",
    [(n_node, n_input, n_input2, bias, name)
        for n_node in n_nodes
        for n_input in n_inputs
        for n_input2 in n_inputs2
        for bias in biases
        for name in names]
)
def test_bilinear_should_throw_value_error(
        n_nodes, n_inputs, n_inputs2, bias, name):
    with pytest.raises(ValueError) as ex:
        x = Bilinear(
                n_nodes=n_nodes, n1_features=n_inputs,
                n2_features=n_inputs2, bias=bias, name=name)

n_nodes = [6, 3]
n_inputs = [6, 5, None]
n_inputs2 = [6, 5, None]
biases = [True, False]
names = ["Test", None]

@pytest.mark.parametrize(
    "n_nodes, n_inputs, n_inputs2, bias, name",
    [(n_node, n_input, n_input2, bias, name)
        for n_node in n_nodes
        for n_input in n_inputs
        for n_input2 in n_inputs2
        for bias in biases
        for name in names]
)
def test_bilinear_get_layer_method(
        n_nodes, n_inputs, n_inputs2, bias, name):

    x = Bilinear(
            n_nodes=n_nodes, n1_features=n_inputs,
            n2_features=n_inputs2, bias=bias, name=name)

    prev_dim = 6

    if n_inputs is None:
        x.get_input_dim(prev_dim)

    details = x.get_layer()

    assert isinstance(details, dict) == True

    if n_inputs:
        assert details['n_inputs'] == n_inputs
    else:
        assert details['n_inputs'] == prev_dim

    assert details["n_nodes"] == n_nodes

    assert details["name"] == name

    assert issubclass(details["layer"], BiLinear) == True

    assert isinstance(details["keyword_arguments"], dict) == True

    if n_inputs:
        assert details["keyword_arguments"]["in_features1"] == n_inputs
        assert details["keyword_arguments"]["in_features2"] == n_inputs2

    else:
        assert details["keyword_arguments"]["in_features1"] == prev_dim
        assert details["keyword_arguments"]["in_features2"] == n_inputs2

    assert details["keyword_arguments"]["out_features"] == n_nodes

    assert details["keyword_arguments"]["bias"] == bias
