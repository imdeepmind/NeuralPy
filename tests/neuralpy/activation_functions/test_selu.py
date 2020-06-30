import pytest
from torch.nn import SELU as _SELU
from neuralpy.activation_functions import SELU

# Possible values
names = [False, 10, 4.5, -5]


@pytest.mark.parametrize(
    "name",
    [(name) for name in names]
)
def test_selu_should_throw_value_error_Exception(name):
    with pytest.raises(ValueError) as ex:
        x = SELU(name=name)

# Possible values
names = ["test1", "test2"]


@pytest.mark.parametrize(
    "name",
    [(name) for name in names]
)
def test_selu_get_input_dim_and_get_layer_method(name):
    x = SELU(name=name)

    assert x.get_input_dim(23, "dense") == None

    details = x.get_layer()

    assert isinstance(details, dict) == True

    assert details["layer_details"] == None

    assert details["name"] == name

    assert issubclass(details["layer"], _SELU) == True

    assert isinstance(details["keyword_arguments"], dict) == True

    assert details["keyword_arguments"]["inplace"] == False
