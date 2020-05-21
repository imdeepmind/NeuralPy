from collections import OrderedDict
from torch import nn, no_grad, device, torch, tensor
from torch.cuda import is_available
from numpy import array

from .utils import is_valid_layer, is_valid_optimizer, is_valid_loss_function

class Sequential():
	def __init__(self, force_cpu=False, training_device=None):
		# Initializing some attributes that we need to function
		self.__layers = []
		self.__model = None
		self.__build = False
		self.__optimizer = None
		self.__loss_function = None
		self.__metrics = None

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

			# Stroing the layer details
			layer_name = layer_details["name"]
			layer_type = layer_details["type"]
			layer_nodes = layer_details["n_nodes"]
			layer_arguments = layer_details["keyword_arguments"]

			# Here we are just storing the ref, not the initialized layer 
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

	def compile(self, optimizer, loss_function, metrics=None):
		# To compile a model, first we need to build it, building it first
		if not self.__build:
			# Calling build
			self.build()

		# Checking the optimizer using the method is_valid_optimizer
		if not is_valid_optimizer(optimizer):
			raise ValueError("Please provide a value neuralpy optimizer")

		# Checking the loss_function using the method is_valid_loss_function
		if not is_valid_loss_function(loss_function):
			raise ValueError("Please provide a value neuralpy loss function")

		# Setting metrics
		self.__metrics = metrics

		# Getting the details of the optimizer using get_optimizer method
		optimizer_details = optimizer.get_optimizer()

		# Getting the details of the loss_function using get_loss_function method
		loss_function_details = loss_function.get_loss_function()

		# Stroing the optimizer details
		optimizer_ref = optimizer_details["optimizer"]
		optimizer_arguments = optimizer_details["keyword_arguments"]

		# Stroing the loss_function details
		loss_function_ref = loss_function_details["loss_function"]
		loss_function_arguments = loss_function_details["keyword_arguments"]

		# Cheking the optimizer_arguments, if it is not None then passing it to the optimizer
		if optimizer_arguments:
			# Initializing the optimizer with optimizer_arguments and models parameters
			optimizer = optimizer_ref(**optimizer_arguments, params=self.__model.parameters())
		else:
			# Initializing the optimizer with models parameters only
			optimizer = optimizer_ref(params=self.__model.parameters())	

		# Checking the loss_function_arguments, if not None and passing it to the loss function
		if loss_function_arguments:
			# Passing the loss_function_arguments to the loss function
			loss_function = loss_function_ref(**loss_function_arguments)
		else:
			# Not passing the loss_function_arguments to the loss function
			loss_function = loss_function_ref()

		# Storing the loss function and optimizer for future use
		self.__optimizer = optimizer
		self.__loss_function = loss_function

	def fit(self, train_data, test_data, epochs=10, batch_size=32):
		# Ectracting the train and test data from the touples
		X_train, y_train = train_data
		X_test, y_test = test_data

		# If batch_size is there then checking the length and comparing it with the length of training data
		if X_train.shape[0] < batch_size:
			# Batch size can not be greater that train data size
			raise ValueError("Batch size is greater than total number of training samples")

		# If batch_size is there then checking the length and comparing it with the length of training data
		if X_test.shape[0] < batch_size:
			# Batch size can not be greater that test data size
			raise ValueError("Batch size is greater than total number of testing samples")

		# Checking the length of input and output
		if X_train.shape[0] != y_train.shape[0]:
			# length of X and y should be same
			raise ValueError("Length of training Input data and training output data should be same")

		# Checking the length of input and output
		if X_test.shape[0] != y_test.shape[0]:
			# length of X and y should be same
			raise ValueError("Length of testing Input data and testing output data should be same")

		# Conveting the data into pytorch tensor
		X_train = tensor(X_train)
		y_train = tensor(y_train)

		X_test = tensor(X_test)
		y_test = tensor(y_test)

		# Initializing a dict to store the training progress, can be used for viz purposes
		if self.__metrics is not None:
			metrics = ["loss"] + self.__metrics
		else:
			metrics = ["loss"]

		history = {
			'batchwise': {},
			'epochwise': {}
		}

		for matrix in metrics:
			history["batchwise"][f"training_{matrix}"] = []
			history["batchwise"][f"validation_{matrix}"] = []
			history["epochwise"][f"training_{matrix}"] = []
			history["epochwise"][f"validation_{matrix}"] = []

		# Running the epochs
		for epoch in range(epochs):
			# Initializing the loss to 0
			training_loss_score = 0
			validation_loss_score = 0

			correct_training = 0
			correct_val = 0

			# Training model :)
			self.__model.train()

			# Spliting the data into batches
			for i in range(0, len(X_train), batch_size):
				# Making the batches
				batch_X = X_train[i:i+batch_size]
				batch_y = y_train[i:i+batch_size]

				# Moving the batches to device
				batch_X, batch_y = batch_X.to(self.__device), batch_y.to(self.__device)

				# Zero grad
				self.__model.zero_grad()

				# Feeding the data into the model
				outputs = self.__model(batch_X.float())

				# Calculating the loss
				train_loss = self.__loss_function(outputs, batch_y)

				# Training
				train_loss.backward()
				self.__optimizer.step()

				# Storing the loss val, batchwise data
				training_loss_score = train_loss.item()
				history["batchwise"]["training_loss"].append(train_loss.item())

				# Calculating accuracy
				# Checking if accuracy is there in metrics
				# TODO: Need to do it more dynamic way
				if "accuracy" in metrics:
					pred = outputs.argmax(dim=1, keepdim=True)
					corrects = pred.eq(batch_y.view_as(pred)).sum().item()
					correct_training += corrects

				# Printing a friendly message to the console
				message = f"Epoch: {epoch+1}/{epochs} - Batch: {i//batch_size+1}/{batch_size} - Training Loss: {train_loss.item():0.4f}"

				if "accuracy" in metrics:
					message += f" - Training Accuracy: {corrects/batch_size*100}%"

				print(message, end="\r")

			# Evluating model
			self.__model.eval()

			# no grad, no training
			with no_grad():
				# Spliting the data into batches
				for i in range(0, len(X_test), batch_size):
					# Making the batches
					batch_X = X_test[i:i+batch_size]
					batch_y = y_test[i:i+batch_size]

					# Moving the batches to device
					batch_X, batch_y = batch_X.to(self.__device), batch_y.to(self.__device)

					# Feeding the data into the model
					outputs = self.__model(batch_X.float())

					# Calculating the loss
					validation_loss = self.__loss_function(outputs, batch_y)

					# Storing the loss val, batchwise data
					validation_loss_score += validation_loss.item()
					history["batchwise"]["validation_loss"].append(validation_loss.item())

			
			# Calculating the mean val loss score for all batches
			validation_loss_score /= batch_size

			# Added the epochwise value to the history dict
			history["epochwise"]["training_loss"].append(training_loss_score)
			history["epochwise"]["validation_loss"].append(validation_loss_score)

			# Printing a friendly message to the console
			print(f"\nValidation Loss: {validation_loss_score:.4f}")

		# Returning history
		return history
	
	def predict(self, X, batch_size=None):
		# Calling model.eval as we are evaluating the model only
		self.__model.eval()

		# Initializing an empty list to store the predictions
		predictions = []

		# Conveting the input X to pytorch Tensor
		X = tensor(X)

		if batch_size:
			# If batch_size is there then checking the length and comparing it with the length of input
			if X.shape[0] < batch_size:
				# Batch size can not be greater that sample size
				raise ValueError("Batch size is greater than total number of samples")

			# Predicting, so no grad
			with no_grad():
				# Spliting the data into batches
				for i in range(0, len(X), batch_size):
					# Generating the batch from X
					batch_X = X[i:i+batch_size]

					# Feeding the batch into the model for predictions
					outputs = self.__model(batch_X.float())

					# Appending the data into the predictions list
					predictions += outputs.numpy().tolist()
		else:
			# Predicting, so no grad
			with no_grad():
				# Feeding the full data into the model for predictions
				outputs = self.__model(X)

				# Appending the data into the predictions list
				predictions += outputs.numpy().tolist()
		
		# Converting the list to numpy array and returning
		return array(predictions)

	def summary(self):
		# Printing the model summary using pytorch model
		if self.__build:
			# Printing models summary
			print(self.__model)

			# Calculating total number of params
			print("Total Number of Parameters: ", sum(p.numel() for p in self.__model.parameters()))
			
			# Calculating total number of trainable params
			print("Total Number of Trainable Parameters: ", sum(p.numel() for p in self.__model.parameters() if p.requires_grad))
		else:
			raise Exception("You need to build the model first")

	def get_model(self):
		# Returning the pytorch model
		return self.__model

	def set_model(self, model):
		# Checking if model is None
		if model is None:
			raise ValueError("Please provide a valid pytorch model")

		# Saving the model
		self.__model = model
		self.__build = True