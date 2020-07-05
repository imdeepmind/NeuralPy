import pytest
from torch.nn import MaxPool2d as _MaxPool2d
from neuralpy.layers import MaxPool2d


def test_maxpool2d_should_throws_type_error():
    with pytest.raises(TypeError) as ex:
        x = MaxPool2d()


@pytest.mark.parametrize(
    'kernel_size, stride, padding, dilation, return_indices, \
    ceil_mode, name',
    [
        (1, '', 0.8, False, 2, 3.6, True),
        (2, '', 0.2, True, 3, 2.4, False),
        (1, 0.2, '', False, 3, 3.4, True),
        (1, 0.9, '', False, 4, 2.2, True),
        (True, '', 9, 0.9, 3, 4.3, False),
        (2, False, '', 0.9, 1, 3.1, True),
        (False, 12, '', 0.9, 3, 2.1, True)
    ]
)
def test_maxpool2d_throws_value_error(
        kernel_size, stride, padding, dilation,
        return_indices, ceil_mode, name):
            with pytest.raises(ValueError) as ex:
                x = MaxPool2d(
                        kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation,
                        return_indices=return_indices,
                        ceil_mode=ceil_mode, name=name)

# Possible values
kernel_sizes = [1, 3]
strides = [2, 1]
paddings = [1, 0]
dilations = [1, 3]
return_indicess = [True, False]
ceil_modes = [False, True]
names = [None, 'Test']


@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation, return_indices, \
    ceil_mode, name",
    [(kernel_size, stride, padding, dilation, return_indices,\
    ceil_mode, name)
    for kernel_size in kernel_sizes
    for stride in strides
    for padding in paddings
    for dilation in dilations
    for return_indices in return_indicess
    for ceil_mode in ceil_modes
    for name in names]
)
def test_maxpool2d_get_layer_method(
        kernel_size, stride, padding, dilation,
        return_indices, ceil_mode, name):

            x = MaxPool2d(
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation,
                    return_indices=return_indices,
                    ceil_mode=ceil_mode, name=name
            )

            details = x.get_layer()

            assert isinstance(details, dict) == True

            assert details['layer_details'] == kernel_size

            assert details['name'] == name

            assert issubclass(details['layer'], _MaxPool2d) == True

            assert isinstance(
                    details['keyword_arguments'], dict) == True

            assert details['keyword_arguments']['kernel_size'] == kernel_size

            assert details['keyword_arguments']['stride'] == stride

            assert details['keyword_arguments']['padding'] == padding

            assert details['keyword_arguments']['dilation'] == dilation

            assert details[
                    'keyword_arguments']['return_indices'] == return_indices

            assert details['keyword_arguments']['ceil_mode'] == ceil_mode
