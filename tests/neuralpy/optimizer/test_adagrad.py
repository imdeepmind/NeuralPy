import pytest
from torch.optim import Adagrad as _Adagrad
from neuralpy.optimizer import Adagrad


@pytest.mark.parametrize(
    "learning_rate, learning_rate_decay, eps, weight_decay",
    [
        (-6, -6, -6, -0.36),
        ("test", -6, -6, -0.36),
        (.001, -6, -6, -0.36),
        (.001, "invalid", -6, -0.36),
        (.001, .003, "invalid", -0.36),
        (.001, .003, -3, -0.36),
        (.001, .003, .001, -0.36),
        (.001, .003, .001, "invalid")
    ]
)
def test_adagrad_should_throw_value_error(learning_rate, learning_rate_decay, eps, weight_decay):
    with pytest.raises(ValueError) as ex:
        x = Adagrad(learning_rate=learning_rate,
                    learning_rate_decay=learning_rate_decay, eps=eps, weight_decay=weight_decay)


@pytest.mark.parametrize(
    "learning_rate, learning_rate_decay, eps, weight_decay",
    [
        (0.1, .001, .2, .32),
        (0.1, .03, .1, .1)
    ]
)
def test_adagrad_get_layer_method(learning_rate, learning_rate_decay, eps, weight_decay):
    x = Adagrad(learning_rate=learning_rate,
                learning_rate_decay=learning_rate_decay, eps=eps, weight_decay=weight_decay)

    details = x.get_optimizer()

    assert isinstance(details, dict) == True

    assert issubclass(details["optimizer"], _Adagrad) == True

    assert isinstance(details["keyword_arguments"], dict) == True

    assert details["keyword_arguments"]["lr"] == learning_rate

    assert details["keyword_arguments"]["lr_decay"] == learning_rate_decay

    # assert details["keyword_arguments"]["eps"] == eps

    assert details["keyword_arguments"]["weight_decay"] == weight_decay


def test_adagrad_get_layer_method_without_parameters():
    x = Adagrad()

    details = x.get_optimizer()

    assert isinstance(details, dict) == True

    assert issubclass(details["optimizer"], _Adagrad) == True

    assert isinstance(details["keyword_arguments"], dict) == True

    assert details["keyword_arguments"]["lr"] == 0.001

    assert details["keyword_arguments"]["lr_decay"] == 0.0

    # assert details["keyword_arguments"]["eps"] == eps

    assert details["keyword_arguments"]["weight_decay"] == 0.0
