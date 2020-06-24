import pytest
from torch.nn import RNN as _RNN
from neuralpy.layers import RNN 

def test_rnn_should_throws_type_error():
    with pytest.raises(TypeError) as ex:
        x = RNN()

input_sizes = [1, '', 0.8]
hidden_sizes = [2, '', 0.2]
num_layerses = [1, '', 0.8]
non_linearities = ['tanh', 1, False]
biases = [False, True, 2]
batch_firsts = [False, 0.2, 2]
dropouts = [1, None, 0.5]
bidirectionals = [False, True, 2]
names  = [False, 12, True]

@pytest.mark.parametrize(
    "input_size, hidden_size, num_layers, non_linearity,\
    bias, batch_first, dropout, bidirectional,name",
    [(input_size, hidden_size, num_layers, non_linearity,\
    bias, batch_first, dropouts, bidirectional,name)
    for input_size in input_sizes
    for hidden_size in hidden_sizes
    for num_layers in num_layerses
    for non_linearity in non_linearities
    for bias in biases
    for batch_first in batch_firsts
    for dropout in dropouts
    for bidirectional in bidirectionals
    for name in names] 
)
def test_rnn_should_throw_value_error(
    input_size, hidden_size, num_layers, non_linearity,
    bias, batch_first, dropout, bidirectional,name):
        with pytest.raises(ValueError) as ex:
            x = RNN(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, non_linearity=non_linearity,
                bias=bias, batch_first=batch_first, dropout=dropout,
                bidirectional=bidirectional, name=name)