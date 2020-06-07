from neuralpy.layers import Dense
from neuralpy.activation_functions import ReLU
from neuralpy.loss_functions import MSELoss
from neuralpy.models.model_helper import generate_layer_name, is_valid_layer

def test_generate_layer_name():
	assert generate_layer_name('Dense', 1) == 'dense_layer_2'
	assert generate_layer_name('ReLU', 1) == 'relu_layer_2'
	assert generate_layer_name('Softmax', 1) == 'softmax_layer_2'

def test_is_valid_layer():
	assert is_valid_layer(Dense(n_nodes=3)) == True
	assert is_valid_layer(ReLU()) == True
	assert is_valid_layer(MSELoss()) == False