import pytest
from torch.nn import BatchNorm1d as _BatchNorm1d
from neuralpy.layers.normalization import BatchNorm1D


@pytest.mark.parametrize(
    "num_features, eps, momentum, affine, \
    track_running_stats, name",
    [
        (0.3, 1.1, 3.4, True, False, "test"),
        (1, 1, 3.4, True, False, "test"),
        (1, 1.1, 3, True, False, "test"),
        (1, 1.1, 3.1, "invalid", False, "test"),
        (1, 1.1, 3.1, True, "invalid", "test"),
        (1, 1.1, 3.1, True, False, 1),
    ],
)
def test_batchnorm1d_should_throw_value_error(
    num_features, eps, momentum, affine, track_running_stats, name
):
    with pytest.raises(ValueError):
        BatchNorm1D(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            name=name,
        )


# Possible values
num_featuress = [1, 4, None]
epss = [1e-03, 1e-04]
momentums = [0.3, 0.4]
affines = [True, False]
track_running_statss = [False, True]
names = ["Test", None]


@pytest.mark.parametrize(
    "num_features, eps, momentum,\
    affine, track_running_stats, name",
    [
        (num_features, eps, momentum, affine, track_running_stats, name)
        for num_features in num_featuress
        for eps in epss
        for momentum in momentums
        for affine in affines
        for track_running_stats in track_running_statss
        for name in names
    ],
)
def test_batchnorm1d_get_layer_method(
    num_features, eps, momentum, affine, track_running_stats, name
):

    x = BatchNorm1D(
        num_features=num_features,
        eps=eps,
        momentum=momentum,
        affine=affine,
        track_running_stats=track_running_stats,
        name=name,
    )

    prev_dim = (3, 6, (6, 18))

    if num_features is None:
        num_features = x.set_input_dim(prev_dim, "conv1d")

    details = x.get_layer()

    assert isinstance(details, dict) is True

    assert details["layer_details"] is None

    assert details["name"] == name

    assert details["layer"] == _BatchNorm1d

    assert details["type"] == "BatchNorm1D"

    assert isinstance(details["keyword_arguments"], dict) is True

    assert details["keyword_arguments"]["eps"] == eps

    assert details["keyword_arguments"]["momentum"] == momentum

    assert details["keyword_arguments"]["affine"] == affine

    assert details["keyword_arguments"]["track_running_stats"] == track_running_stats
