import pytest
from torch.nn import SELU as _SELU
from neuralpy.layers.activation_functions import SELU

# Possible values

names = [False, 10, 4.5, -5]


@pytest.mark.parametrize("name", [name for name in names])
def test_selu_should_throw_value_error_Exception(name):
    with pytest.raises(ValueError):
        SELU(name=name)


# Possible values


names = ["test1", "test2", None]


@pytest.mark.parametrize("name", [name for name in names])
def test_selu_set_input_dim_and_get_layer_method(name):
    x = SELU(name=name)

    assert x.set_input_dim(23, "dense") is None

    details = x.get_layer()

    assert isinstance(details, dict) is True

    assert details["layer_details"] is None

    assert details["name"] == name

    assert issubclass(details["layer"], _SELU) is True

    assert isinstance(details["keyword_arguments"], dict) is True

    assert details["keyword_arguments"]["inplace"] is False
