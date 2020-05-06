import torch.nn as nn

from collections import OrderedDict


class Sequential():
	__layers = []
	__model = None
	__build = False

	def __init__(self):
		super(Sequential, self).__init__()

	def __generate_layer_name(self, type, index):
		# Generating a unique name for the layer
		return f"{type.lower()}_layer_{index+1}"

	def add(self, layer):
		# If we already built the model, then we can not a new layer
		if (self.__build):
			raise Exception("You have built this model already, you can not make any changes in this model")

		# We need to pass the layer
		if not layer:
			raise Exception("You need to pass a layer")

		# Finally adding the layer for layers array
		self.__layers.append(layer)

	def build(self):
		layers = []
		prev_output_dim = 0

		# Iterating through the layers
		for index, layer_ref in enumerate(self.__layers):
			# Generating n_input if not present
			if prev_output_dim is not 0:
				layer_ref.get_input_dim(prev_output_dim)

			# Getting the details of the layer
			layer_details = layer_ref.get_layer()

			layer_name = layer_details["name"]
			layer_type = layer_details["type"]
			layer_nodes = layer_details["n_nodes"]
			layer_arguments = layer_details["keyword_arguments"]
			layer_function_ref = layer_details["layer"]

			# If layer does not have name, then creating a unique name
			if not layer_name:
				layer_name = self.__generate_layer_name(layer_type, index)

			# Calling a layer function with arguments to make the layer
			layer = layer_function_ref(**layer_arguments) 

			# Appending the layer to layers array
			layers.append((layer_name, layer))

			# Storing the n_nodes to calculate the input dim of next layer
			prev_output_dim = layer_nodes


		# Making the pytorch model using nn.Sequential
		self.__model = nn.Sequential(OrderedDict(layers))

		# Chanding the build status to True, so we can not make any changes
		self.__build = True

	def summary(self):
		# Printing the model summary using pytorch model
		if self.__build:
			print(self.__model)
			print("Total Number of Parameters: ", sum(p.numel() for p in self.__model.parameters()))
			print("Total Number of Trainable Parameters: ", sum(p.numel() for p in self.__model.parameters() if p.requires_grad))
		else:
			raise Exception("You need to build the model first")