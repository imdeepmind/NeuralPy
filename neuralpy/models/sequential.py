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

		self.__model = nn.Sequential(OrderedDict(layers))
		self.__build = True

	def summary(self):
		if self.__build:
			n_params = 0

			for layer_details in self.__layers:
				n_params += layer_details["n_params"]

			print(self.__model)
			print("Total Trainable Parameters: ", n_params)
		else:
			print("You need to build the model first")

		














		

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




