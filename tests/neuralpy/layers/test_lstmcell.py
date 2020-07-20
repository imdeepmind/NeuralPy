import pytest
from torch.nn import LSTMCell as _LSTMCell
from neuralpy.layers import LSTMCell


def test_RNN_should_throws_type_error():
    with pytest.raises(TypeError) as ex:
        x = LSTMCell()


@pytest.mark.parametrize(
    "input_size, hidden_size, bias, name",
    [
        (False, 1, True, 'test'),
        (1, 1, None, 'test'),
        (1, None, False, 'test'),
        (1, 2, 3, 'test'),
        (1, 2, 'invalid', 'test'),
        (1, 2, True, "")
    ]
)
def test_LSTMCell_should_throw_value_error(
        input_size, hidden_size, bias, name):
    
    with pytest.raises(ValueError) as ex:
        x = LSTMCell(
            input_size=input_size, hidden_size=hidden_size,
            bias=bias, name=name
        )


# Possible values
input_sizes = [1, 2, None]
hidden_sizes = [2, 2]
biases = [False, True]
names = ['Test', None]


@pytest.mark.parametrize(
    "input_size, hidden_size, bias, name",
    [
        (input_size, hidden_size, bias, name)
        for input_size in input_sizes
        for hidden_size in hidden_sizes
        for bias in biases
        for name in names
    ]
)
def test_LSTMCell_layer_get_method(
        input_size, hidden_size, bias, name):
    
    x = LSTMCell(
        input_size=input_size, hidden_size=hidden_size,
        bias=bias, name=name)

    prev_dim = (6,)

    if input_size is None:
        x.get_input_dim(prev_dim, "LSTMCell")
    
    details = x.get_layer()

    assert isinstance(details, dict) == True

    assert details['layer_details'] == (hidden_size, )

    assert details['name'] == name

    assert issubclass(details['layer'], _LSTMCell) == True

    assert isinstance(details['keyword_arguments'], dict) == True

    if input_size:
        assert details['keyword_arguments']['input_size'] == input_size
    else:
        assert details['keyword_arguments']['input_size'] == prev_dim[0]

    assert details['keyword_arguments']['hidden_size'] == hidden_size

    assert details['keyword_arguments']['bias'] == bias



def test_LSTMCell_with_invalid_layer():
    with pytest.raises(ValueError) as ex:
        x = LSTMCell(hidden_size=128, input_size=None)

        x.get_input_dim((64, ), "dense")