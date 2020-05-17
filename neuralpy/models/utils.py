def is_valid_layer(layer):
	if not layer:
		return False

	try:
		layer_details = layer.get_layer()

		layer_inputs = layer_details["n_inputs"]
		layer_nodes = layer_details["n_nodes"]

		layer_name = layer_details["name"]
		layer_type = layer_details["type"]
		
		layer_arguments = layer_details["keyword_arguments"]
		layer_function_ref = layer_details["layer"]

		if layer_inputs and not isinstance(layer_inputs, int) and layer_inputs < 1:
			return False

		if not isinstance(layer_nodes, int) and layer_nodes < 1:
			return False

		if layer_name and not isinstance(layer_name, str):
			return False

		if not isinstance(layer_type, str):
			return False

		return True

	except AttributeError:
		return False
	except KeyError:
		return False