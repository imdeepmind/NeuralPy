"""Utility functions for activation function"""

# pylint: disable=too-many-arguments
def get_activation_details(name, layer_type, layer, keyword_arguments):
    """
    	Creates the layer details data for the activation function
    """
    return {
        'layer_details': None,
        'name': name,
        'type': layer_type,
        'layer': layer,
        "keyword_arguments": keyword_arguments
    }

def validate_name_field(name):
    """
        A function that validates the name field
    """
    if name is not None and not (isinstance(name, str) and name):
        raise ValueError("Please provide a valid name")
