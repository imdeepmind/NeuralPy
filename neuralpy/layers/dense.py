from torch.nn import Linear

class Dense:
	def __init__(self, n_nodes, n_inputs=None, bias=True, name=None):
		if not n_nodes or n_nodes <= 0:
			raise ValueError("Please provide a valid n_nodes")

		# if not n_inputs or n_inputs <= 0:
		# 	raise ValueError("Please provide a valid n_inputs")

		if not (bias == True or bias == False):
			raise ValueError("Please provide a valid bias")

		if not name or not (isinstance(name, str) and name is not ""):
			raise ValueError("Please provide a valid name")

		self.__n_inputs = n_inputs
		self.__n_nodes = n_nodes

		self.__bias = bias
		self.__name = name

	def get_input_dim(self, prev_input_dim):
		if not self.__n_inputs:
			self.__n_inputs = prev_input_dim

	def get_layer(self):
		return {
			'n_inputs': self.__n_inputs,
			'n_nodes': self.__n_nodes,
			'n_params': self.__n_nodes * self.__n_inputs + self.__n_nodes,
			'name': self.__name,
			'type': 'Dense',
			'layer': Linear,
			"keyword_arguments": {
				'in_features':self.__n_inputs, 
				'out_features':self.__n_nodes, 
				'bias':self.__bias
			}
		}