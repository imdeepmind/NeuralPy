class SequentialHelper:
	def __init__(self):
		pass

	def __generate_layer_name(self, layer_type, index):
		# Generating a unique name for the layer
		return f"{layer_type.lower()}_layer_{index+1}"

	def _is_valid_layer(self, layer):
		# if the layer is none, returning False
		if not layer:
			return False

		try:
			# Calling the get_layer method to details of the layer
			layer_details = layer.get_layer()

			# Checking the layer_details, it should return a dict
			if not isinstance(layer_details, dict):
				return False

			# Here im checking all the keys of object returned from the get_layer method
			layer_inputs = layer_details["n_inputs"]
			layer_nodes = layer_details["n_nodes"]

			layer_name = layer_details["name"]
			layer_type = layer_details["type"]
			
			layer_arguments = layer_details["keyword_arguments"]
			layer_function_ref = layer_details["layer"]

			# Validating layer_inputs
			if layer_inputs and not isinstance(layer_inputs, int) and layer_inputs < 1:
				return False

			# Validating layer_nodes
			if layer_nodes and not isinstance(layer_nodes, int) and layer_nodes < 1:
				return False

			# Validating layer_name
			if layer_name and not isinstance(layer_name, str):
				return False

			# Validating layer_type
			if not isinstance(layer_type, str):
				return False

			# Checking the layer_arguments, it should return a dict or None
			if layer_arguments and not isinstance(layer_arguments, dict):
				return False

			# Checking the layer_function_ref
			# TODO: We should the check the type of layer_function_ref, whether it is pytorch valid layer or not
			if not layer_function_ref:
				return False

			# All good
			return True

		# If there is some missing atricture in the layer, then returning False
		except AttributeError:
			return False
		# If the layer_details dict does not contains a key that it supposed to have
		except KeyError:
			return False

	def _is_valid_optimizer(self, optimizer):
		# If the optimizer is None returning False
		if not optimizer:
			return False

		try:
			# Calling the get_optimizer method to details of the optimizer
			optimizer_details = optimizer.get_optimizer()

			# Checking the optimizer_details, it should return a dict
			if not isinstance(optimizer_details, dict):
				return False

			# Here im checking all the keys of object returned from the get_optimizer method
			optimizer_arguments = optimizer_details["keyword_arguments"]
			optimizer_function_ref = optimizer_details["optimizer"]

			# Checking the optimizer_arguments, it should return a dict or None
			if optimizer_arguments and not isinstance(optimizer_arguments, dict):
				return False

			# Checking the optimizer_function_ref
			# TODO: We should the check the type of optimizer_function_ref, whether it is pytorch valid optimizer or not
			if not optimizer_function_ref:
				return False

			# All good
			return True

		# If there is some missing atricture in the optimizer, then returning False
		except AttributeError:
			return False
		# If the optimizer_details dict does not contains a key that it supposed to have
		except KeyError:
			return False

	def _is_valid_loss_function(self, loss_function):
		# If the loss_function is None returning False
		if not loss_function:
			return False

		try:
			# Calling the get_loss_function method to details of the loss_function
			loss_function_details = loss_function.get_loss_function()

			# Checking the loss_function_details, it should return a dict
			if not isinstance(loss_function_details, dict):
				return False

			# Here im checking all the keys of object returned from the get_loss_function method
			loss_function_arguments = loss_function_details["keyword_arguments"]
			loss_function_function_ref = loss_function_details["loss_function"]

			# Checking the loss_function_arguments, it should return a dict or None
			if loss_function_arguments and not isinstance(loss_function_arguments, dict):
				return False

			# Checking the loss_function_function_ref
			# TODO: We should the check the type of loss_function_function_ref, whether it is pytorch valid loss_function or not
			if not loss_function_function_ref:
				return False

			# All good
			return True

		# If there is some missing atricture in the loss_function, then returning False
		except AttributeError:
			return False
		# If the loss_function_details dict does not contains a key that it supposed to have
		except KeyError:
			return False

	def _build_layer_from_ref_and_details(self, layer_refs):
		# Storing the layer here to build the Sequentuial layer
		layers = []

		# Strong the output dimension, for the next layer, we need this to calculate the next input layer dim
		prev_output_dim = 0

		# Iterating through the layers
		for index, layer_ref in enumerate(layer_refs):

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

		return layers

	def _build_optimizer_from_ref_and_details(self, optimizer_ref, parameters):
		# Getting the details of the optimizer using get_optimizer method
		optimizer_details = optimizer_ref.get_optimizer()

		# Stroing the optimizer details
		optimizer_func = optimizer_details["optimizer"]
		optimizer_arguments = optimizer_details["keyword_arguments"]

		# Creating a variable for the optimizer
		optimizer = None

		# Cheking the optimizer_arguments, if it is not None then passing it to the optimizer
		if optimizer_arguments:
			# Initializing the optimizer with optimizer_arguments and models parameters
			optimizer = optimizer_func(**optimizer_arguments, params=parameters)
		else:
			# Initializing the optimizer with models parameters only
			optimizer = optimizer_func(params=parameters)

		return optimizer

	def _build_loss_function_from_ref_and_details(self, loss_function_ref):
		# Getting the details of the loss_function using get_loss_function method
		loss_function_details = loss_function_ref.get_loss_function()

		# Stroing the loss_function details
		loss_function_func = loss_function_details["loss_function"]
		loss_function_arguments = loss_function_details["keyword_arguments"]

		# Creating a variable for the loss function
		loss_function = None
		
		# Checking the loss_function_arguments, if not None and passing it to the loss function
		if loss_function_arguments:
			# Passing the loss_function_arguments to the loss function
			loss_function = loss_function_func(**loss_function_arguments)
		else:
			# Not passing the loss_function_arguments to the loss function
			loss_function = loss_function_func()

		return loss_function

	def _build_history_object_for_training(self, metrics):
		history = {
			'batchwise': {},
			'epochwise': {}
		}

		for matrix in metrics:
			history["batchwise"][f"training_{matrix}"] = []
			history["batchwise"][f"validation_{matrix}"] = []
			history["epochwise"][f"training_{matrix}"] = []
			history["epochwise"][f"validation_{matrix}"] = []

		return history

	