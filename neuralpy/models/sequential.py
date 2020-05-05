import pickle
import torch.nn as nn
from collections import OrderedDict


class Sequential():
	__layers = []
	__model = None
	__build = False

	def __init__(self):
		super(Sequential, self).__init__()

	def add(self, layer):
		self.__layers.append(layer.get_layer())

	def build(self):
		layers = []

		for layer_details in self.__layers:
			layer = layer_details["layer"](**layer_details["keyword_arguments"])

			layers.append((layer_details["name"], layer))

		model = nn.Sequential(OrderedDict(layers))
		self.__model = model


	def forward(self, x):
		output = None

		for layer_details in self.__layers:
			layer = layer_details["layer"](**layer_details["keyword_arguments"])

			output = layer(x)	

			if layer_details["activation"] is not None:
				output = layer_details["activation"](output)

		return output

	def summary(self):
		lines = "========================================================================\n"
		n_params = 0

		for layer in self.__layers:
			n_params += layer["n_params"]


			lines += f"Layer Name: {layer['name']}	Inputs: {layer['n_inputs']}	Outputs: {layer['n_nodes']}	Parameters: {layer['n_params']}\n"
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

	def compile(self, optimizer, loss_function):
		self.__optimizer = optimizer
		self.__loss_function = loss_function

	def evaluate(self, X, y):
		y_pred = self.predict(X)

		loss = self.__loss_function(y, y_pred)

		return loss




