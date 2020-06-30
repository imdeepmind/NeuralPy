import pytest
from torch.optim import SGD as _SGD
from neuralpy.optimizer import SGD

# Possible values that are invalid
learning_rates = [-6, False, ""]
momentums = [-6, False, ""]
dampenings = ["asd", False, 3]
weight_decays = [-0.36, 'asd', '', False]
nesteroves = [122, ""]


@pytest.mark.parametrize(
    "learning_rate, momentum, dampening, weight_decay, nesterov",
    [
        (-6, -6, False, False, 122),
        ("invalid", -6, False, False, 122),
        (0.00, -6, False, False, 122),
        (0.001, -6, False, False, 122),
        (0.001, False, False, False, 122),
        (0.001, 0.1, False, False, 122),
        (0.001, 0.002, "invalid", False, 122),
        (0.001, 0.002, 0.376, False, 122),
        (0.001, 0.002, 0.342, "test", 122),
        (0.001, 0.002, 0.342, 0.1, 122),
        (0.001, 0.002, 0.342, 0.1, "invalid"),
    ]
)
def test_sgd_should_throw_value_error(learning_rate, momentum, dampening, weight_decay, nesterov):
    with pytest.raises(ValueError) as ex:
        x = SGD(learning_rate=learning_rate, momentum=momentum,
                dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)


# Possible values that are valid
learning_rates = [0.1, 0.002]
momentums = [0.1, 0.002]
dampenings = [0.35]
weight_decays = [0.1, 0.002]
nesteroves = [False, True]


@pytest.mark.parametrize(
    "learning_rate, momentum, dampening, weight_decay, nesterov",
    [(learning_rate, momentum, dampening, weight_decay, nesterov) for learning_rate in learning_rates
     for momentum in momentums
     for dampening in dampenings
     for weight_decay in weight_decays
     for nesterov in nesteroves]
)
def test_sgd_get_layer_method(learning_rate, momentum, dampening, weight_decay, nesterov):
    x = SGD(learning_rate=learning_rate, momentum=momentum,
            dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)

    details = x.get_optimizer()

    assert isinstance(details, dict) == True

    assert issubclass(details["optimizer"], _SGD) == True

    assert isinstance(details["keyword_arguments"], dict) == True

    assert details["keyword_arguments"]["lr"] == learning_rate

    assert details["keyword_arguments"]["momentum"] == momentum

    assert details["keyword_arguments"]["dampening"] == dampening

    assert details["keyword_arguments"]["weight_decay"] == weight_decay

    assert details["keyword_arguments"]["nesterov"] == nesterov

def test_sgd_get_layer_method_without_parameter():
    x = SGD()

    details = x.get_optimizer()

    assert isinstance(details, dict) == True

    assert issubclass(details["optimizer"], _SGD) == True

    assert isinstance(details["keyword_arguments"], dict) == True

    assert details["keyword_arguments"]["lr"] == 0.001

    assert details["keyword_arguments"]["momentum"] == 0.0

    assert details["keyword_arguments"]["dampening"] == 0.0

    assert details["keyword_arguments"]["weight_decay"] == 0.0

    assert details["keyword_arguments"]["nesterov"] == False
