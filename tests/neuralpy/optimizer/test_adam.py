import pytest
from torch.optim import Adam as _Adam
from neuralpy.optimizer import Adam

# Possible values that are invalid
learning_rates = [-6 , False, ""]
betas = [("", 1), ("", ""), (False, 2)]
epses = [-6 , False, ""]
weight_decays = [-0.36, 'asd', '', False]
amsgrads = [12, "", 30.326]

@pytest.mark.parametrize(
	"learning_rate, beta, eps, weight_decay, amsgrad", 
	[
		(-6, ("", 1), False, False, 12),
		(False, ("", 1), False, False, 12),
		(0.001, ("", 1), False, False, 12),
		(0.001, (.3, False), False, False, 12),
		(0.001, (0.10, 2.0), False, False, 12),
		(0.001, (0.10, 2.0), "Invalid", False, 12),
		(0.001, (0.10, 2.0), .2, False, 12),
		(0.001, (0.10, 2.0), .2, "test", 12),
		(0.001, (0.10, 2.0), .2, .32, 12),
		(0.001, (0.10, 2.0), .2, .32, "INVALID")
	]
)
def test_adam_should_throw_value_error(learning_rate, beta, eps, weight_decay, amsgrad):
    with pytest.raises(ValueError) as ex:
        x = Adam(learning_rate=learning_rate, betas=beta, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

# Possible values that are valid
learning_rates = [0.0, 0.1]
betas = [(0.2, 1.0)]
epses = [0.2, 1.0]
weight_decays = [0.32]
amsgrads = [False, True]

@pytest.mark.parametrize(
	"learning_rate, beta, eps, weight_decay, amsgrad", 
	[(learning_rate, beta, eps, weight_decay, amsgrad) for learning_rate in learning_rates
														   for beta in betas
														   for eps in epses
														   for weight_decay in weight_decays
												           for amsgrad in amsgrads]
)
def test_adam_get_layer_method(learning_rate, beta, eps, weight_decay, amsgrad):
	x = Adam(learning_rate=learning_rate, betas=beta, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
		
	details = x.get_optimizer()

	assert isinstance(details, dict) == True

	assert issubclass(details["optimizer"], _Adam) == True

	assert isinstance(details["keyword_arguments"], dict) == True

	assert details["keyword_arguments"]["lr"] == learning_rate

	assert details["keyword_arguments"]["betas"] == beta

	assert details["keyword_arguments"]["eps"] == eps

	assert details["keyword_arguments"]["weight_decay"] == weight_decay

	assert details["keyword_arguments"]["amsgrad"] == amsgrad