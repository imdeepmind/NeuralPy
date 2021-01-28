import pytest
from torch.nn import AvgPool1d as _AvgPool1D
from neuralpy.layers.pooling import AvgPool1D


def test_AvgPool1D_should_throws_type_error():
    with pytest.raises(TypeError):
        AvgPool1D()


@pytest.mark.parametrize(
    "kernel_size, stride, padding, ceil_mode, count_include_pad, name",
    [
        # Checking kernel size validation
        ("invalid", 3, 3, False, False, "test"),
        (("",), 3, 3, False, False, "test"),
        # Checking stride validation
        ((3,), "invalid", 3, False, False, "test"),
        ((3,), ("",), 3, False, False, "test"),
        # Checking padding validation
        ((3,), (3,), "invalid", False, False, "test"),
        ((3,), (3,), ("",), False, False, "test"),
        # Checking ceil mode validation
        ((3,), (3,), (3,), "invalid", False, "test"),
        ((3,), (3,), (3,), 12.5, False, "test"),
        # Checking count_include_pad validation
        ((3,), (3,), (3,), False, "test", "test"),
        ((3,), (3,), (3,), False, 23.4, "test"),
        # Checking name validation
        ((3,), (3,), (3,), False, False, False),
        ((3,), (3,), (3,), False, False, 12),
    ],
)
def test_AvgPool1D_throws_value_error(
    kernel_size, stride, padding, ceil_mode, count_include_pad, name
):
    with pytest.raises(ValueError):
        AvgPool1D(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            name=name,
        )


# Possible values
kernel_sizes = [3, (1,)]
strides = [2, (2,), None]
paddings = [1, (1,)]
ceil_modes = [True]
count_include_pads = [False]
names = [None, "Test"]


@pytest.mark.parametrize(
    "kernel_size, stride, padding, ceil_mode, count_include_pad, name",
    [
        (kernel_size, stride, padding, ceil_mode, count_include_pad, name)
        for kernel_size in kernel_sizes
        for stride in strides
        for padding in paddings
        for ceil_mode in ceil_modes
        for count_include_pad in count_include_pads
        for name in names
    ],
)
def test_AvgPool1D_get_layer_method(
    kernel_size, stride, padding, ceil_mode, count_include_pad, name
):
    x = AvgPool1D(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        name=name,
    )

    prev_dim = (3, 6, (6, 18))

    x.set_input_dim(prev_dim, "conv1d")

    details = x.get_layer()

    assert isinstance(details, dict) is True

    assert details["name"] == name

    assert issubclass(details["layer"], _AvgPool1D) is True

    assert isinstance(details["keyword_arguments"], dict) is True

    assert details["keyword_arguments"]["kernel_size"] == kernel_size

    if stride is None:
        assert details["keyword_arguments"]["stride"] == kernel_size
    else:
        assert details["keyword_arguments"]["stride"] == stride

    assert details["keyword_arguments"]["padding"] == padding

    assert details["keyword_arguments"]["ceil_mode"] == ceil_mode

    assert details["keyword_arguments"]["count_include_pad"] == count_include_pad
