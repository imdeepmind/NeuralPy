import pytest
from torch.nn import Conv1d
from neuralpy.layers import Conv1D


def test_conv1d_should_throw_type_error():
    with pytest.raises(TypeError) as ex:
        x = Conv1D()


@pytest.mark.parametrize(
    "filters, kernel_size, input_shape, stride, padding, dilation, groups, bias, name",
    [
        (0.3, 0.3, 0.36, "invalid", "invalid", "invalid", "groups", False, ""),
        (0.3, 0.3, 0.36, "invalid", "invalid", "invalid", "groups", False, ""),
        (16, 2, 0.36, "invalid", "invalid", "invalid", "groups", False, ""),
        (32, 3, 0.36, "invalid", "invalid", "invalid", "groups", False, ""),
        (32, 3, ("", 26, 26), "invalid", "invalid", "invalid", "groups", False, ""),
        (32, 3, (3, "", 26), "invalid", "invalid", "invalid", "groups", False, ""),
        (32, 3, (3, 26, ""), "invalid", "invalid", "invalid", "groups", False, ""),
        (64, 4, (3, 26, 26), "invalid", "invalid", "invalid", "groups", False, ""),
        (128, 4, (3, 26, 26), 4.6, "invalid", "invalid", "groups", False, ""),
        (256, 4, (3, 26, 26), (34,), "invalid", "invalid", "groups", False, ""),
        (256, 4, (3, 26, 26), (34,), "invalid", "invalid", "groups", False, ""),
        (256, 4, (3, 26, 26), (34,), 7.5, "invalid", "groups", False, ""),
        (256, 4, (3, 26, 26), (34,), (34,), "invalid", "groups", False, ""),
        (256, 4, (3, 26, 26), (34,), (34,), 4.7, "groups", False, ""),
        (256, 4, (3, 26, 26), (34,), (34,), (34,), "groups", False, ""),
        (256, 4, (3, 26, 26), (34,), (34,), (34,), False, False, ""),
        (256, 4, (3, 26, 26), (34,), (34,), (34,), 1, "invalid", ""),
        (256, 4, (3, 26, 26), (34,), (34,), (34,), 1, False, ""),
        (256, 4, (3, 26, 26), (34,), (34,), (34,), 1, False, 34),
    ]
)
def test_conv1d_should_throw_value_error(filters, kernel_size, input_shape, stride, padding, dilation, groups, bias, name):
    with pytest.raises(ValueError) as ex:
        x = Conv1D(filters=filters, kernel_size=kernel_size, input_shape=input_shape,
                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, name=name)


# Possible values
filters = [23, 64]
kernel_size = [4]
input_shape = [(3, 26, 26)]
stride = [(34,)]
padding = [(34,)]
dilation = [(34,)]
groups = 1
biases = [True, False]
names = ["Test", None]


# @pytest.mark.parametrize(
#     "filters, kernel_size, input_shape, stride, padding, dilation, groups, bias, name",
#     [(filters, kernel_size, input_shape, stride, padding, dilation, groups, bias, name)
#      for _filters in filters
#      for _kernel_size in kernel_size
#      for _input_shape in input_shape
#      for _stride in stride
#      for _padding in padding
#      for _dilation in dilation
#      for _groups in groups
#      for _bias in biases
#      for _name in names]
# )
# def test_conv1d_get_layer_method(filters, kernel_size, input_shape, stride, padding, dilation, groups, bias, name):
#     x = Conv1D(filters=filters, kernel_size=kernel_size, input_shape=input_shape,
#                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, name=name)
#     prev_dim = (3, 26, 26),

#     if input_shape is None:
#         x.get_input_dim(prev_dim, "conv1d")

#     details = x.get_layer()

#     assert isinstance(details, dict) == True

#     assert details["layer_details"] == (n_nodes,)

#     assert details["name"] == name

#     assert issubclass(details["layer"], Linear) == True

#     assert isinstance(details["keyword_arguments"], dict) == True

#     if n_inputs:
#         assert details["keyword_arguments"]["in_features"] == n_inputs
#     else:
#         assert details["keyword_arguments"]["in_features"] == prev_dim[0]

#     assert details["keyword_arguments"]["out_features"] == n_nodes

#     assert details["keyword_arguments"]["bias"] == bias
