from torch.nn import LeakyReLU as _LeakyReLU

class LeakyReLU:
	def __init__(self, negative_slope=0.01, name=None):
		# Checking the name field, this is an optional field, if not provided generates a unique name for the layer
		if name is not None and not (isinstance(name, str) and name is not ""):
			raise ValueError("Please provide a valid name")

		self.__name = name
		self.__negative_slope = negative_slope

	def get_input_dim(self, prev_input_dim):
		# Tanh does not need to n_input, so returning None
		return None

	def get_layer(self):
		# Returning all the details of the layer
		return {
			'n_inputs': None,
			'n_nodes': None,
			'name': self.__name,
			'type': 'Tanh',
			'layer': _Tanh,
			"keyword_arguments": {
				'negative_slope': self.__negative_slope
			}
		}
