from neuralpy.layers import Dense
from neuralpy.activation_functions import ReLU
from neuralpy.loss_functions import MSELoss
from neuralpy.models.model_helper import generate_layer_name

def test_generate_layer_name():
	assert generate_layer_name('Dense', 1) == 'dense_layer_2'
	assert generate_layer_name('ReLU', 1) == 'relu_layer_2'
	assert generate_layer_name('Softmax', 1) == 'softmax_layer_2'