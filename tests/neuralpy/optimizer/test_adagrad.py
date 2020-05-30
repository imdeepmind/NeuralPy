import pytest
from torch.optim import Adagrad as _Adagrad
from neuralpy.optimizer import Adagrad

# Possible values that are invalid
learning_rates = [-6 , False, ""]
learning_rate_decays = [-6 , False, ""]
epses = [-6 , False, ""]
weight_decays = [-0.36, 'asd', '', False]

@pytest.mark.parametrize(
	"learning_rate, learning_rate_decay, eps, weight_decay", 
	[(learning_rate, learning_rate_decay, eps, weight_decay) for learning_rate in learning_rates
														   for learning_rate_decay in learning_rate_decays
														   for eps in epses
														   for weight_decay in weight_decays]
)
def test_adam_should_throw_value_error(learning_rate, learning_rate_decay, eps, weight_decay):
    with pytest.raises(ValueError) as ex:
        x = Adagrad(learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, eps=eps, weight_decay=weight_decay)

# Possible values that are valid
learning_rates = [0.0, 0.1]
learning_rate_decays = [0.0, 0.1]
epses = [0.2, 1.0]
weight_decays = [0.32]

@pytest.mark.parametrize(
	"learning_rate, learning_rate_decay, eps, weight_decay", 
	[(learning_rate, learning_rate_decay, eps, weight_decay) for learning_rate in learning_rates
														   for learning_rate_decay in learning_rate_decays
														   for eps in epses
														   for weight_decay in weight_decays]
)
def test_adam_get_layer_method(learning_rate, learning_rate_decay, eps, weight_decay):
	x = Adagrad(learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, eps=eps, weight_decay=weight_decay)
		
	details = x.get_optimizer()

	assert isinstance(details, dict) == True

	assert issubclass(details["optimizer"], _Adagrad) == True

	assert isinstance(details["keyword_arguments"], dict) == True

	assert details["keyword_arguments"]["lr"] == learning_rate

	assert details["keyword_arguments"]["lr_decay"] == learning_rate_decay

	assert details["keyword_arguments"]["eps"] == eps

	assert details["keyword_arguments"]["weight_decay"] == weight_decay