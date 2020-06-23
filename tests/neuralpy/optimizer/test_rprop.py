import pytest
from torch.optim import Rprop as _Rprop
from neuralpy.optimizer import Rprop

@pytest.mark.parametrize(
    "learning_rate, etas, step_sizes",
    [
        (-6, ("", 1.0), ("", 1)),
        (False, ("", 1.0), ("", 1)),
        (0.001, ("", 1.0), ("", 1)),
        (0.001, ("", 1.0), ("", 1)),
        (0.001, (False, 1.0), ("", 1)),
        (0.001, (0.1, "invalid"), ("", 1)),
        (0.001, (0.1, False), ("", 1)),
        (0.001, (0.1, 0.002), ("", 1)),
        (0.001, (0.1, 0.002), (False, 1)),
        (0.001, (0.1, 0.002), (0.001, "invalid")),
        (0.001, (0.1, 0.002), (0.001, True))
    ]
)
def test_rprop_should_throw_value_error(
        learning_rate, etas, step_sizes):
        with pytest.raises(ValueError) as ex:
            x = Rprop(
                learning_rate=learning_rate, etas=etas, step_sizes=step_sizes)

# Possible values that are valid
learning_rates = [0.001, 0.1]
etases = [(0.6, 1.7)]
step_sizeses = [(0.2, 1.0)]

@pytest.mark.parametrize(
    "learning_rate, etas, step_sizes",
    [(learning_rate, etas, step_sizes)
        for learning_rate in learning_rates
        for etas in etases
        for step_sizes in step_sizeses]
)
def test_rprop_get_optimizer_method(learning_rate, etas, step_sizes):
    x = Rprop(
            learning_rate=learning_rate, etas=etas,
            step_sizes=step_sizes)

    details = x.get_optimizer()

    assert isinstance(details, dict) == True

    assert issubclass(details["optimizer"], _Rprop) == True

    assert isinstance(details["keyword_arguments"], dict) == True

    assert details["keyword_arguments"]["lr"] == learning_rate

    assert details["keyword_arguments"]["etas"] == etas

    assert details["keyword_arguments"]["step_sizes"] == step_sizes

def test_rprop_get_optimizer_method_without_parameter():
    x = Rprop()

    details = x.get_optimizer()

    assert isinstance(details, dict) == True

    assert issubclass(details["optimizer"], _Rprop) == True

    assert isinstance(details["keyword_arguments"], dict) == True

    assert details["keyword_arguments"]["lr"] == 0.01

    assert details["keyword_arguments"]["etas"] == (0.5, 1.2)

    assert details["keyword_arguments"]["step_sizes"] == (1e-06, 50.0)