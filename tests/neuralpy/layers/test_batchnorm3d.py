import pytest
from torch.nn import BatchNorm3d as _BatchNorm3d
from neuralpy.layers import BatchNorm3d


def test_batchnorm3d_should_throw_type_error():
    with pytest.raises(TypeError) as ex:
        x = BatchNorm3d()

@pytest.mark.parametrize(
	"num_features, eps, momentum, affine, \
    track_running_status, name", 
	[
		(0.3, 1, "invalid", 2, 3, None),
		(1, "invalid", False, 0.1, 0.3, 0.2),
		(1, 0.4, False, 0.1, 3, True),
		(1, 0.4, 0.3, 1, "invalid", False),
		(1, 0.4, 0.3, "invalid", True, 3),
		(1, 0.4, 0.3, False, True, ""),
		(1, 1e-04, 0.3, True, False, ""),
	]
)
def test_batchnorm3d_should_throw_value_error(
        num_features, eps, momentum, affine,
        track_running_status, name):
    with pytest.raises(ValueError) as ex:
        x = BatchNorm3d(
            num_features=num_features, eps=eps, momentum=momentum,
            affine=affine, track_running_status=track_running_status,
            name=name
        )

# Possible values
num_featuress = [1,4]
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
def test_batchnorm3d_get_layer_method(
        num_features, eps, momentum, affine,
        track_running_status, name):

        x = BatchNorm3d(
            num_features=num_features, eps=eps, momentum=momentum,
            affine=affine, track_running_status=track_running_status,
            name=name)
        
        prev_dim = (6,)

        details = x.get_layer()

        assert isinstance(details, dict) == True

        assert details["layer_details"] == num_features

        assert details["name"] == name

        assert details["layer"] == _BatchNorm3d

        assert details["type"] == "BatchNorm3d"

        assert isinstance(details["keyword_arguments"], dict) == True

        assert details["keyword_arguments"]["eps"] == eps

        assert details["keyword_arguments"]["momentum"] == momentum

        assert details["keyword_arguments"]["affine"] == affine

        assert details["keyword_arguments"]["track_running_status"] == track_running_status
