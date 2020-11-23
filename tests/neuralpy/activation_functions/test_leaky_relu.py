import pytest
from torch.nn import LeakyReLU as _LeakyReLU
from neuralpy.activation_functions import LeakyReLU


@pytest.mark.parametrize('negative_slope, name', [
    ('invalid', 'activation_function'),
    (12, 'activation_function'),
    (False, 'activation_function'),
    (.33, False),
    (.33, 12),
    (.33, 3.6),
    (.33, -2),
])
def test_leaky_relu_should_throw_value_error_Exception(negative_slope, name):
    with pytest.raises(ValueError):
        LeakyReLU(negative_slope=negative_slope, name=name)


# Possible values


negative_slopes = [0.01, 2.342]
names = ['test1', 'test2', None]


@pytest.mark.parametrize('negative_slope, name', [(negative_slope,
                                                   name) for negative_slope in negative_slopes
                                                  for name in names])
def test_leaky_relu_get_input_dim_and_get_layer_method(negative_slope, name):
    x = LeakyReLU(negative_slope=negative_slope, name=name)

    assert x.get_input_dim(12, 'dense') is None

    details = x.get_layer()

    assert isinstance(details, dict) is True

    assert details['layer_details'] is None

    assert details['name'] == name

    assert issubclass(details['layer'], _LeakyReLU) is True

    assert isinstance(details['keyword_arguments'], dict) is True

    assert details['keyword_arguments']['negative_slope'] \
        == negative_slope

    assert details['keyword_arguments']['inplace'] is False
