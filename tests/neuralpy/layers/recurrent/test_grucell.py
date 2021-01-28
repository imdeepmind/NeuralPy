import pytest
from torch.nn import GRUCell as _GRUCell
from neuralpy.layers.recurrent import GRUCell


def test_RNN_should_throws_type_error():
    with pytest.raises(TypeError):
        GRUCell()


@pytest.mark.parametrize(
    "input_size, hidden_size, bias, name",
    [
        (False, 1, True, "test"),
        (1, 1, None, "test"),
        (1, None, False, "test"),
        (1, 2, 3, "test"),
        (1, 2, "invalid", "test"),
        (1, 2, True, ""),
    ],
)
def test_GRUCell_should_throw_value_error(input_size, hidden_size, bias, name):

    with pytest.raises(ValueError):
        GRUCell(input_size=input_size, hidden_size=hidden_size, bias=bias, name=name)


# Possible values
input_sizes = [1, 2, None]
hidden_sizes = [2, 2]
biases = [False, True]
names = ["Test", None]


@pytest.mark.parametrize(
    "input_size, hidden_size, bias, name",
    [
        (input_size, hidden_size, bias, name)
        for input_size in input_sizes
        for hidden_size in hidden_sizes
        for bias in biases
        for name in names
    ],
)
def test_GRUCell_layer_get_method(input_size, hidden_size, bias, name):

    x = GRUCell(input_size=input_size, hidden_size=hidden_size, bias=bias, name=name)

    prev_dim = (6,)

    if input_size is None:
        x.set_input_dim(prev_dim, "GRUCell")

    details = x.get_layer()

    assert isinstance(details, dict) is True

    assert details["layer_details"] == (hidden_size,)

    assert details["name"] == name

    assert issubclass(details["layer"], _GRUCell) is True

    assert isinstance(details["keyword_arguments"], dict) is True

    if input_size:
        assert details["keyword_arguments"]["input_size"] == input_size
    else:
        assert details["keyword_arguments"]["input_size"] == prev_dim[0]

    assert details["keyword_arguments"]["hidden_size"] == hidden_size

    assert details["keyword_arguments"]["bias"] == bias


def test_GRUCell_with_invalid_layer():
    with pytest.raises(ValueError):
        x = GRUCell(hidden_size=128, input_size=None)

        x.set_input_dim((64,), "conv1d")
