from torch.nn import Linear

class Dense:
	def __init__(self, n_nodes, n_inputs=None, bias=True, name=None, activation=None):
		self.__n_inputs = n_inputs
		self.__n_nodes = n_nodes

		self.__bias = bias
		self.__name = name
		
		self.__activation = activation

	def get_layer(self):
		return {
			'n_inputs': self.__n_inputs,
			'n_nodes': self.__n_nodes,
			'n_params': self.__n_nodes * self.__n_inputs + self.__n_nodes,
			'activation': self.__activation,
			'name': self.__name,
			'type': 'Dense',
			'layer': Linear,
			"keyword_arguments": {
				'in_features':self.__n_inputs, 
				'out_features':self.__n_nodes, 
				'bias':self.__bias
			}
		}