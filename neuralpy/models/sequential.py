import pickle

class Sequential:
	__layers = []

	def __init__(self):
		pass

	def add(self, layer):
		self.__layers.append(layer)

	def summary(self):
		lines = "========================================================================\n"
		n_params = 0

		for layer in self.__layers:
			layer_details = layer.get_details()
			n_params += layer_details["n_params"]


			lines += f"Layer Name: {layer_details['name']}	Inputs: {layer_details['n_inputs']}	Outputs: {layer_details['n_nodes']}	Parameters: {layer_details['n_params']}\n"
			lines += "========================================================================\n"

		lines += f"\nTotal Trainable Parameters: {n_params}\n"

		return lines

	def predict(self, x):
		output = None

		for layer in self.__layers:
			if output is None:
				output = layer.forward(x)
			else:
				output = layer.forward(output)


		return output

	def save(self, path):
		with open(path, "wb") as f:
			pickle.dump(self.__layers, f)

	def load_model(self, path):
		with open(path, "rb") as f:
			self.__layers = pickle.load(f)



