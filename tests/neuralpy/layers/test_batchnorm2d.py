import pytest
from torch.nn import BatchNorm2d as _BatchNorm2d
from neuralpy.layers import BatchNorm2D


def test_batchnorm2d_should_throw_type_error():
    with pytest.raises(TypeError) as ex:
        x = BatchNorm2D()

@pytest.mark.parametrize(
	"num_features, eps, momentum, affine, \
    track_running_status, name", 
	[
		(0.3, 1.1, 3.4, True, False, "test"),
        (1, 1, 3.4, True, False, "test"),
        (1, 1.1, 3, True, False, "test"),
        (1, 1.1, 3.1, "invalid", False, "test"),
        (1, 1.1, 3.1, True, "invalid", "test"),
        (1, 1.1, 3.1, True, False, 1)
	]
)
def test_batchnorm3d_should_throw_value_error(
        num_features, eps, momentum, affine,
        track_running_status, name):
    with pytest.raises(ValueError) as ex:
        x = BatchNorm2D(
            num_features=num_features, eps=eps, momentum=momentum,
            affine=affine, track_running_status=track_running_status,
            name=name
        )

# Possible values
num_featuress = [1, 4, None]
epss = [1e-03, 1e-04]
momentums = [0.3, 0.4]
affines = [True, False]
track_running_statuss = [False, True]
names = ["Test", None]

@pytest.mark.parametrize(
    "num_features, eps, momentum,\
    affine, track_running_status, name",
    [(
        num_features, eps, momentum, affine,
        track_running_status, name)
        for num_features in num_featuress
        for eps in epss
        for momentum in momentums
        for affine in affines
        for track_running_status in track_running_statuss
        for name in names]
)
def test_batchnorm2d_get_layer_method(
        num_features, eps, momentum, affine,
        track_running_status, name):

        x = BatchNorm2D(
            num_features=num_features, eps=eps, momentum=momentum,
            affine=affine, track_running_status=track_running_status,
            name=name)
        
        prev_dim = (3, 6, (6, 18, 32, 32))

        if num_features is None:

             num_features = x.get_input_dim(prev_dim, "conv2d")

        details = x.get_layer()

        assert isinstance(details, dict) == True

        assert details["layer_details"] == (num_features, )

        assert details["name"] == name

        assert details["layer"] == _BatchNorm2d

        assert details["type"] == "BatchNorm2d"

        assert isinstance(details["keyword_arguments"], dict) == True

        assert details["keyword_arguments"]["eps"] == eps

        assert details["keyword_arguments"]["momentum"] == momentum

        assert details["keyword_arguments"]["affine"] == affine

        assert details["keyword_arguments"]["track_running_status"] == track_running_status
