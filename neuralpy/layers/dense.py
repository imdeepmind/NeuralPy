from torch.nn import Linear

class Dense:
	def __init__(self, n_nodes, n_inputs=None, bias=True, name=None):
		# Checking the n_nodes field
		if not n_nodes or not isinstance(n_nodes, int) or n_nodes <= 0:
			raise ValueError("Please provide a valid n_nodes")

		# Checking the n_input field, it is a optional field
		if n_inputs and isinstance(n_inputs, int) and n_inputs <= 0:
			raise ValueError("Please provide a valid n_inputs")

		# Checking the bias field, this is also optional, default to True
		if not isinstance(bias, bool):
			raise ValueError("Please provide a valid bias")

		# Checking the name field, this is an optional field, if not provided generates a unique name for the layer
		if name and not (isinstance(name, str) and name is not ""):
			raise ValueError("Please provide a valid name")

		# Storing the data
		self.__n_inputs = n_inputs
		self.__n_nodes = n_nodes

		self.__bias = bias
		self.__name = name

	def get_input_dim(self, prev_input_dim):
		# Checking if n_inputs is there or not, not overwriting the n_input field
		if not self.__n_inputs:
			self.__n_inputs = prev_input_dim

	def get_layer(self):
		# Returning all the details of the layer
		return {
			'n_inputs': self.__n_inputs,
			'n_nodes': self.__n_nodes,
			'name': self.__name,
			'type': 'Dense',
			'layer': Linear,
			"keyword_arguments": {
				'in_features':self.__n_inputs, 
				'out_features':self.__n_nodes, 
				'bias':self.__bias
			}
		}