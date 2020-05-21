def is_valid_layer(layer):
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

def is_valid_optimizer(optimizer):
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

def is_valid_loss_function(loss_function):
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