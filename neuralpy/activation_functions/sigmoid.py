from torch.nn import Sigmoid as _Sigmoid

class Sigmoid:
	def __init__(self, name=None):
		# Checking the name field,
		# this is an optional field, if not provided generates a unique name for the activation function
		if name is not None and not (isinstance(name, str) and name is not ""):
			raise ValueError("Please provide a valid name")

		self.__name = name

	def get_input_dim(self, prev_input_dim):
		# Sigmoid does not need to n_input, so returning None
		return None

	def get_layer(self):
		# Returning all the details of the activation function
		return {
			'n_inputs': None,
			'n_nodes': None,
			'name': self.__name,
			'type': 'Sigmoid',
			'layer': _Sigmoid,
			"keyword_arguments": None
		}
