import pytest
from torch.optim import SGD as _SGD
from neuralpy.optimizer import SGD

# Possible values that are invalid
learning_rates = [-6 , False, ""]
momentums = [-6 , False, ""]
weight_decays = [-0.36, 'asd', '', False]

@pytest.mark.parametrize(
	"learning_rate, momentum, weight_decay", 
	[(learning_rate, momentum, weight_decay) for learning_rate in learning_rates
									         for momentum in momentums
									         for weight_decay in weight_decays]
)
def test_sgd_should_throw_value_error(learning_rate, momentum, weight_decay):
    with pytest.raises(ValueError) as ex:
        x = SGD(learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay)


# Possible values that are valid
learning_rates = [0.1, 0.002]
momentums = [0.1, 0.002]
weight_decays = [0.1, 0.002]

@pytest.mark.parametrize(
	"learning_rate, momentum, weight_decay", 
	[(learning_rate, momentum, weight_decay) for learning_rate in learning_rates
									         for momentum in momentums
									         for weight_decay in weight_decays]
)
def test_sgd_get_layer_method(learning_rate, momentum, weight_decay):
	x = SGD(learning_rate=learning_rate, momentum=momentum, dampening=0, weight_decay=weight_decay, nesterov=False)
		
	details = x.get_optimizer()

	assert isinstance(details, dict) == True

	assert issubclass(details["optimizer"], _SGD) == True

	assert isinstance(details["keyword_arguments"], dict) == True

	assert details["keyword_arguments"]["lr"] == learning_rate

	assert details["keyword_arguments"]["momentum"] == momentum

	assert details["keyword_arguments"]["dampening"] == 0

	assert details["keyword_arguments"]["weight_decay"] == weight_decay

	assert details["keyword_arguments"]["nesterov"] == False