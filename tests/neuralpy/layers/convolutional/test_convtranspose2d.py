import pytest
from torch.nn import ConvTranspose2d as _ConvTranspose2d
from neuralpy.layers.convolutional import ConvTranspose2d


def test_convtranspose2d_should_throw_type_error():
    with pytest.raises(TypeError):
        ConvTranspose2d()


@pytest.mark.parametrize(
    "in_channels, out_channels, kernel_size, input_shape, stride, padding,\
    output_padding, groups, bias, dilation,name",
    [
        # Checking in_channels validation
        (
            0.3, 0.3, 0.36, "invalid", "invalid",
            "invalid", "invalid", "groups", False, "invalid", ""
        ),
        # Checking out_channels validation
        (
            3, 0.3, 0.36, "invalid", "invalid",
            "invalid", "invalid", "groups", False, "invalid", ""
        ),
        # Checking kernel_size validation
        (
            3, 3, 0.36, "invalid", "invalid",
            "invalid", "invalid", "groups", False, "invalid", ""
        ),
        # Checking input_shape validation
        (
            3, 3, 5, "invalid", "invalid",
            "invalid", "invalid", "groups", False, "invalid", ""
        ),
        (
            3, 3, 5, (-1,), "invalid",
            "invalid", "invalid", "groups", False, "invalid", ""
        ),
        (
            3, 3, 5, (3, 0.6), "invalid",
            "invalid", "invalid", "groups", False, "invalid", ""
        ),
        # Checking stride validation
        (
            3, 3, 3, (3, 3), 3.5,
            "invalid", "invalid", "groups", False, "invalid", ""
        ),
        # Checking padding validation
        (
            3, 3, 3, (3, 3), (4, 4),
            "invalid", "invalid", "groups", False, "invalid", ""
        ),
        # Checking out_padding validation
        (
            3, 3, 3, (3, 3), (4, 4),
            4, "invalid", "groups", False, "invalid", ""
        ),
        # Checking group validation
        (
            3, 3, 3, (3, 3), (4, 4),
            4, (5, 5), "groups", False, "invalid", ""
        ),
        # Checking bias validation
        (
            3, 3, 3, (3, 3), (4, 4),
            4, (5, 5), 4, "invalid", "invalid", ""
        ),
        # Checking dilation validation
        (
            3, 3, 3, (3, 3), (4, 4),
            4, (5, 5), 4, True, (3, 0.6), ""
        ),
        (
            3, 3, 3, (3, 3), (4, 4),
            4, (5, 5), 4, True, "invalid", ""
        ),
        # Checking name validation
        (
            3, 3, 3, (3, 3), (4, 4),
            4, (5, 5), 4, True, (3, 3), ""
        ),
        (
            3, 3, 3, (3, 3), (4, 4),
            4, (5, 5), 4, True, (3, 3), True
        )
    ]
)
def test_convtrans2d_should_throw_value_error(
        in_channels, out_channels, kernel_size, input_shape, stride,
        padding, output_padding, groups, bias, dilation, name):
    with pytest.raises(ValueError):
        ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, input_shape=input_shape,
            stride=stride, padding=padding,
            output_padding=output_padding, groups=groups,
            bias=bias, dilation=dilation, name=name)


in_channels = [4]
out_channels = [6]
kernel_size = [3, (3, 3)]
input_shape = [(4, 5), None]
stride = [3, (3, 3)]
padding = [3, (3, 3)]
output_padding = [2, (2, 2)]
groups = [1]
bias = [True]
dilation = [3, (3, 3)]
name = ["Name", None]


@pytest.mark.parametrize(
    "_in_channels, _out_channels, _kernel_size, _input_shape, _stride, \
    _padding, _out_padding, _groups, _bias, _dilation, _name",
    [(_in_channels, _out_channels, _kernel_size, _input_shape, _stride,
      _padding, _out_padding, _groups, _bias, _dilation, _name)
     for _in_channels in in_channels
     for _out_channels in out_channels
     for _kernel_size in kernel_size
     for _input_shape in input_shape
     for _stride in stride
     for _padding in stride
     for _out_padding in output_padding
     for _groups in groups
     for _bias in bias
     for _dilation in dilation
     for _name in name]
)
def test_convtrans2d_get_layer_method(
        _in_channels, _out_channels, _kernel_size, _input_shape, _stride,
        _padding, _out_padding, _groups, _bias, _dilation, _name):
    x = ConvTranspose2d(in_channels=_in_channels, out_channels=_out_channels,
                        kernel_size=_kernel_size, input_shape=_input_shape,
                        stride=_stride, padding=_padding, output_padding=_out_padding,
                        groups=_groups, bias=_bias, dilation=_dilation, name=_name)

    if _input_shape is None:
        prev_dim = (3, 3 * 32, (3, 32))
        x.get_input_dim(prev_dim, "ConvTranspose2d")

    details = x.get_layer()

    assert isinstance(details, dict) is True

    assert details["name"] == _name

    assert issubclass(details["layer"], _ConvTranspose2d) is True

    assert details["type"] == "ConvTranspose2d"

    assert isinstance(details["keyword_arguments"], dict) is True


def test_convtrans2d_get_layer_method_invalid_layer():
    x = ConvTranspose2d(
        in_channels=16, out_channels=12, kernel_size=2, input_shape=None)

    prev_dim = (3, 3 * 32, (3, 32))

    with pytest.raises(ValueError):
        x.get_input_dim(prev_dim, "conv2d")
