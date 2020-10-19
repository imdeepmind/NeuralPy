import inspect


class CustomLayer:
  def __init__(self, layer_class, layer_type, layer_name):
    # TODO: In future, we might need to add more checks to
    # confirm this is a PyTorch valid class
    if not inspect.isclass(layer_class):
      raise ValueError("Please provide a valid layer class")

    if not (isinstance(layer_type, str) and layer_type):
      raise ValueError("Please provide a valid layer type")

    if layer_name is not None and not (isinstance(layer_name, str) and layer_name):
      raise ValueError("Please provide a valid name")

    self.__layer_type = layer_type
    self.__layer = layer_class
    self.__layer_name = layer_name

  def _get_layer_details(self, layer_details=None, keyword_arguments=None):
    """
        Creates the layer details data for the activation function
    """
    return {
        'layer_details': layer_details,
        'name': self.__layer_name,
        'type': self.__layer_type,
        'layer': self.__layer,
        "keyword_arguments": keyword_arguments
    }

  def _get_layer_class(self):
    return self.__layer
  
  def _get_layer_type(self):
    return self.__layer_type
  
  def _get_layer_name(self):
    return self.__layer_name