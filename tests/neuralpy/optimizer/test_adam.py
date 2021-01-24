import pytest
from torch.optim import Adam as _Adam
from neuralpy.optimizer import Adam


@pytest.mark.parametrize(
    "learning_rate, beta, eps, weight_decay, amsgrad",
    [
        (-6, (0.9, 0.999), 1e-08, 0.0, False),
        (False, (0.9, 0.999), 1e-08, 0.0, False),
        (0.001, False, 1e-08, 0.0, False),
        (0.001, ("", 0.3), 1e-08, 0.0, False),
        (0.001, (-3, 0.3), 1e-08, 0.0, False),
        (0.001, (0.3, False), 1e-08, 0.0, False),
        (0.001, (0.3, "invalid"), 1e-08, 0.0, False),
        (0.001, (0.10, 2.0), False, 0.0, False),
        (0.001, (0.10, 2.0), "Invalid", 0.0, False),
        (0.001, (0.10, 2.0), -0.3, 0.0, False),
        (0.001, (0.10, 2.0), 0.2, False, False),
        (0.001, (0.10, 2.0), 0.2, "test", False),
        (0.001, (0.10, 2.0), 0.2, 0.32, 12),
        (0.001, (0.10, 2.0), 0.2, 0.32, "INVALID"),
    ],
)
def test_adam_should_throw_value_error(learning_rate, beta, eps, weight_decay, amsgrad):
    with pytest.raises(ValueError):
        Adam(
            learning_rate=learning_rate,
            betas=beta,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )


# Possible values that are valid
learning_rates = [0.0, 0.1]
betas = [(0.2, 1.0)]
epses = [0.2, 1.0]
weight_decays = [0.32]
amsgrads = [False, True]


@pytest.mark.parametrize(
    "learning_rate, beta, eps, weight_decay, amsgrad",
    [
        (learning_rate, beta, eps, weight_decay, amsgrad)
        for learning_rate in learning_rates
        for beta in betas
        for eps in epses
        for weight_decay in weight_decays
        for amsgrad in amsgrads
    ],
)
def test_adam_get_layer_method(learning_rate, beta, eps, weight_decay, amsgrad):
    x = Adam(
        learning_rate=learning_rate,
        betas=beta,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
    )

    details = x.get_optimizer()

    assert isinstance(details, dict) is True

    assert issubclass(details["optimizer"], _Adam) is True

    assert isinstance(details["keyword_arguments"], dict) is True

    assert details["keyword_arguments"]["lr"] == learning_rate

    assert details["keyword_arguments"]["betas"] == beta

    assert details["keyword_arguments"]["eps"] == eps

    assert details["keyword_arguments"]["weight_decay"] == weight_decay

    assert details["keyword_arguments"]["amsgrad"] == amsgrad
