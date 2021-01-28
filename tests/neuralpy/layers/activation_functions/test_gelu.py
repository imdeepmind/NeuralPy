import pytest
from torch.nn import GELU as _GELU
from neuralpy.layers.activation_functions import GELU

# Possible values

names = [False, 12, 3.6, -2]


@pytest.mark.parametrize("name", [name for name in names])
def test_gelu_should_throw_value_error_Exception(name):
    with pytest.raises(ValueError):
        GELU(name=name)


# Possible values


names = ["test1", "test2", None]


@pytest.mark.parametrize("name", [name for name in names])
def test_gelu_set_input_dim_and_get_layer_method(name):
    x = GELU(name=name)

    assert x.set_input_dim(12, "dense") is None

    details = x.get_layer()

    assert isinstance(details, dict) is True

    assert details["layer_details"] is None

    assert details["name"] == name

    assert issubclass(details["layer"], _GELU) is True

    assert details["keyword_arguments"] is None
