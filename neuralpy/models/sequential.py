from collections import OrderedDict
from torch import nn, no_grad, device, torch, tensor
from torch.cuda import is_available
from numpy import array

from .sequential_helper import SequentialHelper

class Sequential(SequentialHelper):
	def __init__(self, force_cpu=False, training_device=None):
		super(Sequential, self).__init__()
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

	def add(self, layer):
		# If we already built the model, then we can not a new layer
		if (self.__build):
			raise Exception("You have built this model already, you can not make any changes in this model")

		# Layer verification using the method is_valid_layer
		if not self._is_valid_layer(layer):
			raise ValueError("Please provide a valid neuralpy layer")

		# Finally adding the layer for layers array
		self.__layers.append(layer)

	def build(self):
		# Building the layers from the layer refs and details
		layers = self._build_layer_from_ref_and_details(self.__layers)

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
		if not self._is_valid_optimizer(optimizer):
			raise ValueError("Please provide a value neuralpy optimizer")

		# Checking the loss_function using the method is_valid_loss_function
		if not self._is_valid_loss_function(loss_function):
			raise ValueError("Please provide a value neuralpy loss function")

		# Setting metrics
		self.__metrics = metrics
		
		# Storing the loss function and optimizer for future use
		self.__optimizer = self._build_optimizer_from_ref_and_details(optimizer, self.__model.parameters())
		self.__loss_function = self._build_loss_function_from_ref_and_details(loss_function)

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
		metrics = []
		
		if self.__metrics is not None:
			metrics = ["loss"] + self.__metrics
		else:
			metrics = ["loss"]

		# Building the history object
		history = self._build_history_object_for_training(metrics)

		# Running the epochs
		for epoch in range(epochs):
			# Initializing the loss and accuracy with 0
			training_loss_score = 0
			validation_loss_score = 0

			correct_training = 0
			correct_val = 0

			# Training model :)
			self.__model.train()

			# Spliting the data into batches
			for i in range(0, len(X_train), batch_size):
				# Making the batches
				batch_X = X_train[i:i+batch_size].float()
				if "accuracy" in metrics:
					batch_y = y_train[i:i+batch_size]
				else:
					batch_y = y_train[i:i+batch_size].float()	
				

				# Moving the batches to device
				batch_X, batch_y = batch_X.to(self.__device), batch_y.to(self.__device)

				# Zero grad
				self.__model.zero_grad()

				# Feeding the data into the model
				outputs = self.__model(batch_X)

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
					corrects = self._calculate_accuracy(batch_y, outputs)

					correct_training += corrects

					history["batchwise"]["training_accuracy"].append(corrects/batch_size*100)

					self._print_training_progress(epoch, epochs, i, batch_size, len(X_train), train_loss.item(), corrects)
				else:
					self._print_training_progress(epoch, epochs, i, batch_size, len(X_train), train_loss.item())

			# Evluating model
			self.__model.eval()

			# no grad, no training
			with no_grad():
				# Spliting the data into batches
				for i in range(0, len(X_test), batch_size):
					# Making the batches
					batch_X = X_train[i:i+batch_size].float()
					if "accuracy" in metrics:
						batch_y = y_train[i:i+batch_size]
					else:
						batch_y = y_train[i:i+batch_size].float()	

					# Moving the batches to device
					batch_X, batch_y = batch_X.to(self.__device), batch_y.to(self.__device)

					# Feeding the data into the model
					outputs = self.__model(batch_X)

					# Calculating the loss
					validation_loss = self.__loss_function(outputs, batch_y)

					# Storing the loss val, batchwise data
					validation_loss_score += validation_loss.item()
					history["batchwise"]["validation_loss"].append(validation_loss.item())

					# Calculating accuracy
					# Checking if accuracy is there in metrics
					# TODO: Need to do it more dynamic way
					if "accuracy" in metrics:
						corrects = corrects = self._calculate_accuracy(batch_y, outputs)

						correct_val += corrects

						history["batchwise"]["validation_accuracy"].append(corrects/batch_size*100)
			
			# Calculating the mean val loss score for all batches
			validation_loss_score /= batch_size

			# Added the epochwise value to the history dict
			history["epochwise"]["training_loss"].append(training_loss_score)
			history["epochwise"]["validation_loss"].append(validation_loss_score)

			# Checking if accuracy is there in metrics
			if "accuracy" in metrics:
				# Adding data into hostory dict
				history["epochwise"]["training_accuracy"].append(correct_training/len(X_train)*100)
				history["epochwise"]["training_accuracy"].append(correct_val/len(X_test)*100)

				# Printing a friendly message to the console
				self._print_validation_progress(validation_loss_score, len(X_train), correct_val)
			else:
				# Printing a friendly message to the console
				self._print_validation_progress(validation_loss_score, len(X_train))
				

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
					batch_X = X[i:i+batch_size].float()

					# Feeding the batch into the model for predictions
					outputs = self.__model(batch_X)

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

	def predict_classes(self, X, batch_size=None):
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
					batch_X = X[i:i+batch_size].float()

					# Feeding the batch into the model for predictions
					outputs = self.__model(batch_X)

					# Predicting the class
					pred = outputs.argmax(dim=1, keepdim=True)

					# Appending the data into the predictions list
					predictions += pred.numpy().tolist()
		else:
			# Predicting, so no grad
			with no_grad():
				# Feeding the full data into the model for predictions
				outputs = self.__model(X)

				# Predicting the class
				pred = outputs.argmax(dim=1, keepdim=True)

				# Appending the data into the predictions list
				predictions += pred.numpy().tolist()
		
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