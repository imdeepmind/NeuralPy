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
		if not isinstance(layer_nodes, int) and layer_nodes < 1:
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