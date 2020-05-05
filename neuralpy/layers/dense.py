from torch.nn import Linear

class Dense:
	def __init__(self, n_inputs, n_nodes, bias=True, name=None, activation=None):
		return {
			'n_inputs': n_inputs,
			'n_nodes': n_nodes,
			'activation': activation,
			'name': name,
			'type': 'Dense',
			'layer': Linear(in_deatures=n_inputs, out_features=n_nodes, bias=bias)
		}