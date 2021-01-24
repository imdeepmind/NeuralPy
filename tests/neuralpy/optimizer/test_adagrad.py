import pytest
from torch.optim import Adagrad as _Adagrad
from neuralpy.optimizer import Adagrad


@pytest.mark.parametrize(
    "learning_rate, learning_rate_decay, eps, weight_decay",
    [
        (-6, 0.003, 0.003, 0.003),
        ("test", 0.003, 0.003, 0.003),
        (0.001, -6, 0.003, 0.003),
        (0.001, "invalid", 0.003, 0.003),
        (0.001, 0.003, "invalid", 0.003),
        (0.001, 0.003, -3, 0.003),
        (0.001, 0.003, 0.001, -0.36),
        (0.001, 0.003, 0.001, "invalid"),
    ],
)
def test_adagrad_should_throw_value_error(
    learning_rate, learning_rate_decay, eps, weight_decay
):
    with pytest.raises(ValueError):
        Adagrad(
            learning_rate=learning_rate,
            learning_rate_decay=learning_rate_decay,
            eps=eps,
            weight_decay=weight_decay,
        )


@pytest.mark.parametrize(
    "learning_rate, learning_rate_decay, eps, weight_decay",
    [(0.1, 0.001, 0.2, 0.32), (0.1, 0.03, 0.1, 0.1)],
)
def test_adagrad_get_layer_method(
    learning_rate, learning_rate_decay, eps, weight_decay
):
    x = Adagrad(
        learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay,
        eps=eps,
        weight_decay=weight_decay,
    )

    details = x.get_optimizer()

    assert isinstance(details, dict) is True

    assert issubclass(details["optimizer"], _Adagrad) is True

    assert isinstance(details["keyword_arguments"], dict) is True

    assert details["keyword_arguments"]["lr"] == learning_rate

    assert details["keyword_arguments"]["lr_decay"] == learning_rate_decay

    # assert details["keyword_arguments"]["eps"] == eps

    assert details["keyword_arguments"]["weight_decay"] == weight_decay


def test_adagrad_get_layer_method_without_parameters():
    x = Adagrad()

    details = x.get_optimizer()

    assert isinstance(details, dict) is True

    assert issubclass(details["optimizer"], _Adagrad) is True

    assert isinstance(details["keyword_arguments"], dict) is True

    assert details["keyword_arguments"]["lr"] == 0.001

    assert details["keyword_arguments"]["lr_decay"] == 0.0

    # assert details["keyword_arguments"]["eps"] == eps

    assert details["keyword_arguments"]["weight_decay"] == 0.0
