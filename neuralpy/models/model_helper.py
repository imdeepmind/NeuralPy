"""Sequential Helper functions"""


def generate_layer_name(layer_type, index):
    """
    Generates a unique layer name
    """
    # Generating a unique name for the layer
    return f"{layer_type.lower()}_layer_{index+1}"


def build_layer_from_dict(layer_refs):
    """
    Builds model from layers ref and details
    """
    # Storing the layer here to build the Sequential layer
    layers = []

    # Strong the output dimension, for the next layer,
    # we need this to calculate the next input layer dim
    prev_layer_details = None
    prev_layer_type = None

    # Iterating through the layers
    for index, layer_ref in enumerate(layer_refs):
        # Generating n_input if not present
        if prev_layer_details and prev_layer_type:
            # For each layer, we have this method that returns the new input layer
            # for next dim based on the previous layer details and type
            # the prev_layer_details is a tuple that contains all the information
            # need for the layer to predict the input shape
            # The prev_layer_type is the type of the layer, based on it,
            # the layers can calculate the input shape
            # for example, in cnn, after the conv layers, when the dense layer need
            # to do some complicated calculations to get the input shape of the Dense
            # layer based on the input shape, stride, padding, etc
            layer_ref.set_input_dim(prev_layer_details, prev_layer_type)

        # Getting the details of the layer using the get_layer method
        layer_details = layer_ref.get_layer()

        # Storing the layer details
        layer_name = layer_details["name"]
        layer_type = layer_details["type"]
        layer_details_info = layer_details["layer_details"]
        layer_arguments = layer_details["keyword_arguments"]

        # Here we are just storing the ref, not the initialized layer
        layer_function_ref = layer_details["layer"]

        # If layer does not have name, then creating a unique name
        if not layer_name:
            # This method generates a unique layer name based on layer type and
            # index
            layer_name = generate_layer_name(layer_type, index)

        # If layer_arguments is not None, then the layer accepts some parameters
        # to initialize
        if layer_arguments is not None:
            # Here passing the layer_arguments to the layer reference to initialize
            # the layer
            layer = layer_function_ref(**layer_arguments)
        else:
            # This layer does not need layer_arguments so not passing anything
            layer = layer_function_ref()

        # Appending the layer to layers array
        layers.append((layer_name, layer))

        if layer_details_info:
            prev_layer_details = layer_details_info
            prev_layer_type = layer_type

    return layers


def build_optimizer_from_dict(optimizer_ref, parameters):
    """
    Builds optimizer from ref and details
    """
    # Getting the details of the optimizer using get_optimizer method
    optimizer_details = optimizer_ref.get_optimizer()

    # Storing the optimizer details
    optimizer_func = optimizer_details["optimizer"]
    optimizer_arguments = optimizer_details["keyword_arguments"]

    # Creating a variable for the optimizer
    optimizer = None

    # Checking the optimizer_arguments, if it is not None then passing it to
    # the optimizer
    if optimizer_arguments:
        # Initializing the optimizer with optimizer_arguments and models
        # parameters
        optimizer = optimizer_func(**optimizer_arguments, params=parameters)
    else:
        # Initializing the optimizer with models parameters only
        optimizer = optimizer_func(params=parameters)

    return optimizer, optimizer_arguments


def build_loss_function_from_dict(loss_function_ref):
    """
    Builds loss function
    """
    # Getting the details of the loss_function using get_loss_function method
    loss_function_details = loss_function_ref.get_loss_function()

    # Storing the loss_function details
    loss_function_func = loss_function_details["loss_function"]
    loss_function_arguments = loss_function_details["keyword_arguments"]

    # Creating a variable for the loss function
    loss_function = None

    # Checking the loss_function_arguments, if not None and passing it to the
    # loss function
    if loss_function_arguments:
        # Passing the loss_function_arguments to the loss function
        loss_function = loss_function_func(**loss_function_arguments)
    else:
        # Not passing the loss_function_arguments to the loss function
        loss_function = loss_function_func()

    return loss_function, loss_function_arguments


def build_history_object(metrics):
    """
    Builds history object
    """
    history = {"batchwise": {}, "epochwise": {}}

    for matrix in metrics:
        history["batchwise"][f"training_{matrix}"] = []
        history["batchwise"][f"validation_{matrix}"] = []
        history["epochwise"][f"training_{matrix}"] = []
        history["epochwise"][f"validation_{matrix}"] = []

    return history


def calculate_accuracy(y, y_pred):
    """
    Calculates accuracy from real labels and predicted labels
    """
    pred = y_pred.argmax(dim=1, keepdim=True)

    corrects = pred.eq(y.view_as(pred)).sum().item()

    return corrects


def print_training_progress(
    epoch, epochs, batch, batches, no_samples, training_loss, training_corrects=None
):
    """
    Show a training progress text
    """
    # Printing a friendly message to the console
    message = (
        f"Epoch: {epoch+1}/{epochs} - "
        f"Batch: {batch//batches+1}/{no_samples//batches} - "
        f"Training Loss: {training_loss:0.4f}"
    )

    if training_corrects:
        message += f" - Training Accuracy: {training_corrects/batches*100:.4f}%"

    print("\r" + message, end="")


def print_validation_progress(validation_loss, no_samples, validation_corrects=None):
    """
    Show a validation progress text
    """
    message = ""
    if validation_corrects:
        message = (
            f"\rValidation Loss: {validation_loss:.4f} - "
            f"Validation Accuracy: {validation_corrects/no_samples*100:.4f}%"
        )
    else:
        if isinstance(validation_loss, (int, float)):
            message = f"\rValidation Loss: {validation_loss:.4f}"
        else:
            message = "\rValidation Loss: NA"

    print("\r" + message, end="")
