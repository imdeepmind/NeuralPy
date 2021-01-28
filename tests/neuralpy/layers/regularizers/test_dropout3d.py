import pytest
from torch.nn import Dropout3d as _Dropout3D
from neuralpy.layers.regularizers import Dropout3D


@pytest.mark.parametrize(
    "p, name", [(6.3, "Test"), (-4.2, "Test"), (0.33, False), (0.56, 12)]
)
def test_dense_should_throw_value_error(p, name):
    with pytest.raises(ValueError):
        Dropout3D(p=p, name=name)


@pytest.mark.parametrize("p, name", [(0.3, "test"), (0.2, None)])
def test_dense_get_layer_method(p, name):
    x = Dropout3D(p=p, name=name)

    assert x.set_input_dim(12, "dense") is None

    details = x.get_layer()

    assert isinstance(details, dict) is True

    assert details["layer_details"] is None

    assert details["name"] == name

    assert issubclass(details["layer"], _Dropout3D) is True

    assert isinstance(details["keyword_arguments"], dict) is True

    assert details["keyword_arguments"]["p"] == p

    assert details["keyword_arguments"]["inplace"] is False


def test_dense_get_layer_method_wit_no_parameter():
    x = Dropout3D()

    assert x.set_input_dim(12, "dense") is None

    details = x.get_layer()

    assert isinstance(details, dict) is True

    assert details["layer_details"] is None

    assert details["name"] is None

    assert issubclass(details["layer"], _Dropout3D) is True

    assert isinstance(details["keyword_arguments"], dict) is True

    assert details["keyword_arguments"]["p"] == 0.5

    assert details["keyword_arguments"]["inplace"] is False
