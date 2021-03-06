import pytest
from torch.nn import Flatten as _Flatten
from neuralpy.layers.other import Flatten


@pytest.mark.parametrize(
    "start_dim, end_dim", [(0.3, False), (1, 0.3), (1, "invalid"), ("invalid", 1)]
)
def test_flatten_should_throw_value_error(start_dim, end_dim):
    with pytest.raises(ValueError):
        Flatten(start_dim=start_dim, end_dim=end_dim)


# possible values
start_dims = [6, 3]
end_dims = [3, 1]


@pytest.mark.parametrize(
    "start_dim, end_dim",
    [(start_dim, end_dim) for start_dim in start_dims for end_dim in end_dims],
)
def test_flatten_get_layer_method(start_dim, end_dim):
    x = Flatten(start_dim=start_dim, end_dim=end_dim)
    prev_input_dim = (6,)

    details = x.get_layer()

    assert isinstance(details, dict) is True

    assert x.set_input_dim(prev_input_dim, "flatten") is None

    if start_dim:
        assert details["keyword_arguments"]["start_dim"] == start_dim
    else:
        assert details["keyword_arguments"]["start_dim"] == 1

    if end_dim:
        assert details["keyword_arguments"]["end_dim"] == end_dim
    else:
        assert details["keyword_arguments"]["end_dim"] == -1

    assert issubclass(details["layer"], _Flatten) is True

    assert isinstance(details["keyword_arguments"], dict) is True

    assert details["keyword_arguments"]["start_dim"] == start_dim

    assert details["keyword_arguments"]["end_dim"] == end_dim
