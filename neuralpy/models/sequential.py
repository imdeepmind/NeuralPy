class Sequential:
	__layers = []

	def __init__(self):
		pass

	def add(self, layer):
		self.__layers.append(layer)

	def summary(self):
		lines = "========================================================================\n"
		n_params = 1
		
		for layer in self.__layers:
			layer_details = layer.get_details()
			n_params += layer_details["n_params"]


			lines += f"Layer Name: {layer_details['name']}	Inputs: {layer_details['n_inputs']}	Nodes: {layer_details['n_nodes']}	Parameters: {layer_details['n_params']}\n"
			lines += "========================================================================\n"

		lines += f"\nTotal Trainable Parameters: {n_params}\n"

		return lines




