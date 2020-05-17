from collections import OrderedDict
from torch import Tensor
from torch import nn
from torch import no_grad
from torch import device
from torch.cuda import is_available

from .utils import is_valid_layer

class Sequential():
	def __init__(self, force_cpu=False, training_device=None):
		# Initializing some attributes that we need to function
		self.__layers = []
		self.__model = None
		self.__build = False
		self.__optimizer = None
		self.__loss_function = None

		# Checking the force_cpu parameter
		if not (force_cpu == True or force_cpu == False):
			raise ValueError(f"You have provided an invalid value for the parameter force_cpu")

		# Checking the training_device parameter and comparing it with pytorch device class
		if training_device and not issubclass(training_device, device):
			raise ValueError("Please provide a valid neuralpy device class")

		# if force_cpu then using CPU
		# if device provided, then using it
		# else auto detecting the device, if cuda available then using it (default option)
		if training_device:
			self.__device = training_device
		elif force_cpu == True:
			self.__device = device("cpu")
		else:
			if is_available():
				self.__device = device("cuda:0") # TODO: currently setting it to cuda:0, may need to change it
			else:
				self.__device = device("cpu")

	def __generate_layer_name(self, layer_type, index):
		# Generating a unique name for the layer
		return f"{layer_type.lower()}_layer_{index+1}"

	def add(self, layer):
		# If we already built the model, then we can not a new layer
		if (self.__build):
			raise Exception("You have built this model already, you can not make any changes in this model")

		# Layer verification using the method is_valid_layer
		if not is_valid_layer(layer):
			raise ValueError("Please provide a valid neuralpy layer")

		# Finally adding the layer for layers array
		self.__layers.append(layer)

	def build(self):
		# Storing the layer here to build the Sequentuial layer
		layers = []

		# Strong the output dimension, for the next layer, we need this to calculate the next input layer dim
		prev_output_dim = 0

		# Iterating through the layers
		for index, layer_ref in enumerate(self.__layers):

			# Generating n_input if not present
			if prev_output_dim is not 0:
				# For each layer, we have this method that returns the new input layer for next dim
				# based on the previous output dim
				layer_ref.get_input_dim(prev_output_dim)

			# Getting the details of the layer using the get_layer method
			layer_details = layer_ref.get_layer()

			# Stroning the layer details
			layer_name = layer_details["name"]
			layer_type = layer_details["type"]
			layer_nodes = layer_details["n_nodes"]
			layer_arguments = layer_details["keyword_arguments"]

			# Here we are just storing the ref, not the initialized the layer 
			layer_function_ref = layer_details["layer"]

			# If layer does not have name, then creating a unique name
			if not layer_name:
				# This method generates a unique layer name based on layer type and index
				layer_name = self.__generate_layer_name(layer_type, index)

			# If layer_arguments is not None, then the layer accepts some parameters to initialize 
			if layer_arguments is not None:
				# Here passing the layer_arguments to the layer reference to initialize the layer
				layer = layer_function_ref(**layer_arguments) 
			else:
				# This layer does not need layer_arguments so not passing anything
				layer = layer_function_ref() 

			# Appending the layer to layers array
			layers.append((layer_name, layer))

			# Checking layer_nodes value against some condition, and then storing the n_nodes to calculate the input dim of next layer 
			if layer_nodes is not None and layer_nodes >= 0:
				prev_output_dim = layer_nodes


			# Making the pytorch model using nn.Sequential
			self.__model = nn.Sequential(OrderedDict(layers))

			# Transferring the model to device
			self.__model.to(self.__device)

			# Printing a message with the device name
			print("The model is running on", self.__device)

		# Chanding the build status to True, so we can not make any changes
		self.__build = True

	def compile(self, optimizer, loss_function):
		# To compile a model, first we need to build it, building it first
		if not self.__build:
			# Calling build
			self.build()

		optimizer_details = optimizer.get_optimizer()
		loss_function_details = loss_function.get_loss_function()

		optimizer_ref = optimizer_details["optimizer"]
		optimizer_arguments = optimizer_details["keyword_arguments"]

		loss_function_ref = loss_function_details["loss_function"]
		loss_function_arguments = loss_function_details["keyword_arguments"]

		optimizer = optimizer_ref(**optimizer_arguments, params=self.__model.parameters())
		loss_function = loss_function_ref(**loss_function_arguments)

		self.__optimizer = optimizer
		self.__loss_function = loss_function


	def fit(self, train_data, test_data, epochs=10, batch_size=32):
		X_train, y_train = train_data
		X_test, y_test = test_data

		X_train = Tensor(X_train)
		y_train = Tensor(y_train)

		X_test = Tensor(X_test)
		y_test = Tensor(y_test)

		history = {
			'batchwise': {
				'training_loss': [],
				'validation_loss': []
			},
			'epochwise': {
				'training_loss': [],
				'validation_loss': []
			}
		}

		for epoch in range(epochs):
			training_loss_score = 0
			validation_loss_score = 0

			self.__model.train()

			for i in range(0, len(X_train), batch_size):
				batch_X = X_train[i:i+batch_size]
				batch_y = y_train[i:i+batch_size]

				batch_X, batch_y = batch_X.to(self.__device), batch_y.to(self.__device)

				self.__model.zero_grad()

				outputs = self.__model(batch_X)
				train_loss = self.__loss_function(outputs, batch_y)

				train_loss.backward()
				self.__optimizer.step()

				training_loss_score = train_loss.item()
				history["batchwise"]["training_loss"].append(train_loss.item())

				print(f"Epoch: {epoch+1}/{epochs} - Batch: {i//batch_size+1}/{batch_size} - Training Loss: {train_loss.item():0.4f}", end="\r")

			self.__model.eval()

			with no_grad():
				for i in range(0, len(X_test), batch_size):
					batch_X = X_test[i:i+batch_size]
					batch_y = y_test[i:i+batch_size]

					batch_X, batch_y = batch_X.to(self.__device), batch_y.to(self.__device)

					outputs = self.__model(batch_X)
					validation_loss = self.__loss_function(outputs, batch_y)

					validation_loss_score += validation_loss.item()
					history["batchwise"]["validation_loss"].append(validation_loss.item())

			

			validation_loss_score /= batch_size

			history["epochwise"]["training_loss"].append(training_loss_score)
			history["epochwise"]["validation_loss"].append(validation_loss_score)

			print(f"\nValidation Loss: {validation_loss_score:.4f}")


		return history
	
	def predict(self, X, batch_size=32):
		predictions = []

		X = Tensor(X)

		with no_grad():
			for i in range(0, len(X), batch_size):
				batch_X = X[i:i+batch_size]

				outputs = self.__model(batch_X)

				predictions += outputs.numpy().tolist()


		return [x[0] for x in predictions]

	def summary(self):
		# Printing the model summary using pytorch model
		if self.__build:
			print(self.__model)
			print("Total Number of Parameters: ", sum(p.numel() for p in self.__model.parameters()))
			print("Total Number of Trainable Parameters: ", sum(p.numel() for p in self.__model.parameters() if p.requires_grad))
		else:
			raise Exception("You need to build the model first")