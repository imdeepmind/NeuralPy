import pytest
from itertools import permutations

from neuralpy.layers import Dense

n_nodes = [0.3, 6.3, -0.36, 'asd', '', False]
n_inputs = [0.3, 6.3, -0.36, 'asd', '', False]
biases = [1, "", 0.3]
names = [False, 12]


def build_options():
    res = [(n_node, n_input, bias, name) for n_node in n_nodes
           for n_input in n_inputs
           for bias in biases
           for name in names]

    return res

def test_dense_exception_no_parameter():
    with pytest.raises(TypeError) as ex:
        x = Dense()


@pytest.mark.parametrize("n_nodes, n_inputs, bias, name", build_options())
def test_dense_exception_with_parameter(n_nodes, n_inputs, bias, name):
    with pytest.raises(ValueError) as ex:
        x = Dense(n_nodes=n_nodes, n_inputs=n_inputs, bias=bias, name=name)
