from torch.nn import ReLU as _ReLU

class ReLU:
	def __init__(self, name=None)
		if not name or not (isinstance(name, str) and name is not ""):
			raise ValueError("Please provide a valid name")

		self.__name = name

	def get_layer(self):
		return {
			'n_inputs': 0,
			'n_nodes': 0,
			'n_params': 0,
			'name': self.__name,
			'type': 'ReLU',
			'layer': _ReLU,
			"keyword_arguments": None
		}