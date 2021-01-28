import pytest
from torch.nn import Softmax as _Softmax
from neuralpy.layers.activation_functions import Softmax


@pytest.mark.parametrize(
    "dim, name",
    [
        ((12, 2), "test"),
        ("asda", "test"),
        (123.23, "test"),
        (2, False),
        (2, 12),
        (2, 3.6),
        (2, -2),
    ],
)
def test_leaky_relu_should_throw_value_error_Exception(dim, name):
    with pytest.raises(ValueError):
        Softmax(dim=dim, name=name)


# Possible values


dims = [1, 2]
names = ["test1", "test2", None]


@pytest.mark.parametrize("dim, name", [(dim, name) for dim in dims for name in names])
def test_leaky_relu_set_input_dim_and_get_layer_method(dim, name):
    x = Softmax(dim=dim, name=name)

    assert x.set_input_dim(12, "dense") is None

    details = x.get_layer()

    assert isinstance(details, dict) is True

    assert details["layer_details"] is None

    assert details["name"] == name

    assert issubclass(details["layer"], _Softmax) is True

    assert isinstance(details["keyword_arguments"], dict) is True

    assert details["keyword_arguments"]["dim"] == dim
