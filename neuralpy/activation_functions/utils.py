"""Utility functions for activation function"""

def get_activation_details(n_inputs, n_nodes, name, layer_type, layer, keyword_arguments):
    """
    	Creates the layer details data for the activation function
    """
    return {
        'n_inputs': n_inputs,
        'n_nodes': n_nodes,
        'name': name,
        'type': layer_type,
        'layer': layer,
        "keyword_arguments": keyword_arguments
    }
