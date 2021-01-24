import pytest
from torch.nn import Conv1d as _Conv1D
from neuralpy.layers.convolutional import Conv1D


def test_conv1d_should_throw_type_error():
    with pytest.raises(TypeError):
        Conv1D()


@pytest.mark.parametrize(
    "filters, kernel_size, input_shape, stride, padding, dilation, groups, bias, name",
    [
        # Checking Filters validation
        (0.3, 0.3, 0.36, "invalid", "invalid", "invalid", "groups", False, ""),
        (0.3, 0.3, 0.36, "invalid", "invalid", "invalid", "groups", False, ""),
        # Chaking kernel size validation
        (16, 2, 0.36, "invalid", "invalid", "invalid", "groups", False, ""),
        (32, False, 0.36, "invalid", "invalid", "invalid", "groups", False, ""),
        (32, "", "", "invalid", "invalid", "invalid", "groups", False, ""),
        (32, ("",), "invalid", "invalid", "invalid", "invalid", "groups", False, ""),
        # Chaking input shape validation
        (64, (3,), "invalid", "invalid", "invalid", "invalid", "groups", False, ""),
        (128, (3,), False, 4.6, "invalid", "invalid", "groups", False, ""),
        (256, (3,), (3, ""), "invalid", "invalid", "invalid", "groups", False, ""),
        (256, (3,), ("", 3), "invalid", "invalid", "invalid", "groups", False, ""),
        # Checking stride validation
        (256, (3,), (3, 3), "invalid", "invalid", "invalid", "groups", False, ""),
        (256, (3,), (3, 3), 4.5, "invalid", "invalid", "groups", False, ""),
        (256, (3,), (3, 3), False, "invalid", "invalid", "groups", False, ""),
        (256, (3,), (3, 3), ("",), "invalid", "invalid", "groups", False, ""),
        (256, (3,), (3, 3), (3.4,), "invalid", "invalid", "groups", False, ""),
        # Checking padding validation
        (256, (3,), (3, 3), (3,), "invalid", "invalid", "groups", False, ""),
        (256, (3,), (3, 3), (3,), False, "invalid", "groups", False, ""),
        (256, (3,), (3, 3), (3,), 6.5, "invalid", "groups", False, ""),
        (256, (3,), (3, 3), (3,), ("",), "invalid", "groups", False, ""),
        # Checking the dilation
        (256, (3,), (3, 3), (3,), (3,), "invalid", "groups", False, ""),
        (256, (3,), (3, 3), (3,), (3,), False, "groups", False, ""),
        (256, (3,), (3, 3), (3,), (3,), 4.5, "groups", False, ""),
        (256, (3,), (3, 3), (3,), (3,), ("",), "groups", False, ""),
        (256, (3,), (3, 3), (3,), (3,), (4.5,), "groups", False, ""),
        # Checking the groups
        (256, (3,), (3, 3), (3,), (3,), (4,), "groups", False, ""),
        (256, (3,), (3, 3), (3,), (3,), (4,), 4.5, False, ""),
        (256, (3,), (3, 3), (3,), (3,), (4,), False, False, ""),
        # Checking the bias
        (256, (3,), (3, 3), (3,), (3,), (4,), 12, 12, ""),
        (256, (3,), (3, 3), (3,), (3,), (4,), 12, "invalid", ""),
        (256, (3,), (3, 3), (3,), (3,), (4,), 12, 0.45, ""),
        # Checking name validation
        (256, (3,), (3, 3), (3,), (3,), (4,), 12, False, ""),
        (256, (3,), (3, 3), (3,), (3,), (4,), 12, False, 12),
    ],
)
def test_conv1d_should_throw_value_error(
    filters, kernel_size, input_shape, stride, padding, dilation, groups, bias, name
):
    with pytest.raises(ValueError):
        Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            input_shape=input_shape,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            name=name,
        )


# Possible values
filters = [23]
kernel_size = [4, (4,)]
input_shape = [(3, 2), None]
stride = [(34,), 34]
padding = [(34,), 34]
dilation = [(34,), 34]
groups = [1]
biases = [True]
names = ["Test", None]


@pytest.mark.parametrize(
    "_filters, _kernel_size, _input_shape, _stride, _padding, _dilation, _groups, \
        _bias, _name",
    [
        (
            _filters,
            _kernel_size,
            _input_shape,
            _stride,
            _padding,
            _dilation,
            _groups,
            _bias,
            _name,
        )
        for _filters in filters
        for _kernel_size in kernel_size
        for _input_shape in input_shape
        for _stride in stride
        for _padding in padding
        for _dilation in dilation
        for _groups in groups
        for _bias in biases
        for _name in names
    ],
)
def test_conv1d_get_layer_method(
    _filters,
    _kernel_size,
    _input_shape,
    _stride,
    _padding,
    _dilation,
    _groups,
    _bias,
    _name,
):
    x = Conv1D(
        filters=_filters,
        kernel_size=_kernel_size,
        input_shape=_input_shape,
        stride=_stride,
        padding=_padding,
        dilation=_dilation,
        groups=_groups,
        bias=_bias,
        name=_name,
    )

    if _input_shape is None:
        prev_dim = (3, 3 * 32, (3, 32))
        x.set_input_dim(prev_dim, "conv1d")

    details = x.get_layer()

    assert isinstance(details, dict) is True

    # TODO: Need to check the formula
    # if input_shape is None:
    # assert details["layer_details"] == (n_nodes,)
    # else:
    # assert details["layer_details"] == (n_nodes,)

    assert details["name"] == _name

    assert issubclass(details["layer"], _Conv1D) is True

    assert details["type"] == "Conv1D"

    assert isinstance(details["keyword_arguments"], dict) is True


def test_conv1d_get_layer_method_invlaid_layer():
    x = Conv1D(filters=32, kernel_size=2, input_shape=None)

    prev_dim = (3, 3 * 32, (3, 32))

    with pytest.raises(ValueError):
        x.set_input_dim(prev_dim, "conv2d")
