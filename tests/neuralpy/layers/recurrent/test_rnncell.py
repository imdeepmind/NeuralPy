import pytest
from torch.nn import RNNCell as _RNNCell
from neuralpy.layers.recurrent import RNNCell


def test_RNN_should_throws_type_error():
    with pytest.raises(TypeError):
        RNNCell()


@pytest.mark.parametrize(
    "input_size, hidden_size, bias, non_linearity, name",
    [
        (False, 1, True, "tanh", "test"),
        (1, 1, None, "tanh", "test"),
        (1, None, False, "tanh", "test"),
        (1, 2, 3, "tanh", "test"),
        (1, 2, "invalid", "tanh", "test"),
        (1, 2, False, "invalid", "test"),
        (1, 2, True, False, None),
        (1, 2, True, "tanh", ""),
    ],
)
def test_RNNCell_should_throw_value_error(
    input_size, hidden_size, bias, non_linearity, name
):

    with pytest.raises(ValueError):
        RNNCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            non_linearity=non_linearity,
            name=name,
        )


# Possible values
input_sizes = [1, 2, None]
hidden_sizes = [2, 2]
biases = [False, True]
non_linearities = ["tanh", "relu"]
names = ["Test", None]


@pytest.mark.parametrize(
    "input_size, hidden_size, bias, non_linearity, name",
    [
        (input_size, hidden_size, bias, non_linearity, name)
        for input_size in input_sizes
        for hidden_size in hidden_sizes
        for bias in biases
        for non_linearity in non_linearities
        for name in names
    ],
)
def test_RNNCell_layer_get_method(input_size, hidden_size, bias, non_linearity, name):

    x = RNNCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=bias,
        non_linearity=non_linearity,
        name=name,
    )

    prev_dim = (6,)

    if input_size is None:
        x.set_input_dim(prev_dim, "RNNCell")

    details = x.get_layer()

    assert isinstance(details, dict) is True

    assert details["layer_details"] == (hidden_size,)

    assert details["name"] == name

    assert issubclass(details["layer"], _RNNCell) is True

    assert isinstance(details["keyword_arguments"], dict) is True

    if input_size:
        assert details["keyword_arguments"]["input_size"] == input_size
    else:
        assert details["keyword_arguments"]["input_size"] == prev_dim[0]

    assert details["keyword_arguments"]["hidden_size"] == hidden_size

    assert details["keyword_arguments"]["bias"] == bias

    assert details["keyword_arguments"]["non_linearity"] == non_linearity


def test_RNNCell_with_invalid_layer():
    with pytest.raises(ValueError):
        x = RNNCell(hidden_size=128, input_size=None)

        x.set_input_dim((64,), "conv1d")
