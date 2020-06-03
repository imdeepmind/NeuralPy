class Model:
	def __init__(self):
		pass

    def __is_valid_optimizer(self, optimizer):
        """
            Checks if optimizer is valid or not
        """
        # If the optimizer is None returning False
        if not optimizer:
            return False

        try:
            # Calling the get_optimizer method to details of the optimizer
            optimizer_details = optimizer.get_optimizer()

            # Checking the optimizer_details, it should return a dict
            if not isinstance(optimizer_details, dict):
                return False

            # checking all the keys of object returned from the get_optimizer method
            optimizer_arguments = optimizer_details["keyword_arguments"]
            optimizer_function_ref = optimizer_details["optimizer"]

            # Checking the optimizer_arguments, it should return a dict or None
            if optimizer_arguments and not isinstance(optimizer_arguments, dict):
                return False

            # Checking the optimizer_function_ref
            if not optimizer_function_ref:
                return False

            # All good
            return True

        # If there is some missing architecture in the optimizer, then returning False
        except AttributeError:
            return False
        # If the optimizer_details dict does not contains a key that it supposed to have
        except KeyError:
            return False

    def __is_valid_loss_function(self, loss_function):
        """
            Checks if a loss function is valid or not
        """
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
            if not loss_function_function_ref:
                return False

            # All good
            return True

        # If there is some missing architecture in the loss_function, then returning False
        except AttributeError:
            return False
        # If the loss_function_details dict does not contains a key that it supposed to have
        except KeyError:
            return False

    def __build_optimizer_from_ref_and_details(self, optimizer_ref):
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

        # Checking the optimizer_arguments, if it is not None then passing it to the optimizer
        if optimizer_arguments:
            # Initializing the optimizer with optimizer_arguments and models parameters
            optimizer = optimizer_func(
                **optimizer_arguments, params=self.__model.parameters())
        else:
            # Initializing the optimizer with models parameters only
            optimizer = optimizer_func(params=self.__model.parameters())

        return optimizer

    def __build_loss_function_from_ref_and_details(self, loss_function_ref):
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

        # Checking the loss_function_arguments, if not None and passing it to the loss function
        if loss_function_arguments:
            # Passing the loss_function_arguments to the loss function
            loss_function = loss_function_func(**loss_function_arguments)
        else:
            # Not passing the loss_function_arguments to the loss function
            loss_function = loss_function_func()

        return loss_function

    def compile(self, optimizer, loss_function, metrics=None):
        """
            In NeuralPy model, compile method is responsible for attaching a loss
            function and optimizer with the model and this method needs to
            be called before training. This method also attaches metrics that needed
            to be calculated during training.

            The .compile() method internally calls the .build(),
            so there is no need to call .build().

            Supported Arguments:
                optimizer: (NeuralPy Optimizer class) Adds a optimizer to the model
                loss_function: (NeuralPy Loss Function class) Adds a loss function to the model
                metrics: ([String]) Metrics that will be evaluated by the model.
                    Currently only supports "accuracy".
        """

        # Checking the optimizer using the method is_valid_optimizer
        if not self.__is_valid_optimizer(optimizer):
            raise ValueError("Please provide a value neuralpy optimizer")

        # Checking the loss_function using the method is_valid_loss_function
        if not self.__is_valid_loss_function(loss_function):
            raise ValueError("Please provide a value neuralpy loss function")

        if metrics and not isinstance(metrics, list):
            raise ValueError("Please provide a valid metrics")

        # Setting metrics
        if metrics:
            self.__metrics = ["loss"] + metrics
        else:
            self.__metrics = ["loss"]

        # Storing the loss function and optimizer for future use
        self.__optimizer = build_optimizer_from_ref_and_details(optimizer)
        self.__loss_function = build_loss_function_from_ref_and_details(loss_function)

    def get_model(self):
        """
            The .get_model() method is used for getting the PyTorch model from the NeuralPy model.
            After extracting the model, the model can be treated just like a regular
            PyTorch model.

            Supported Arguments
                None
        """
        # Returning the PyTorch model
        return self.__model

    def set_model(self, model):
        """
            The .set_model() method is used for converting a PyTorch model to a NeuralPy model.
            After this conversion, the model can be trained using NeuralPy optimizer
            and loss_functions.

            Supported Arguments
                model: (PyTorch model) A valid class based on Sequential PyTorch model.


        """
        # Checking if model is None
        if model is None:
            raise ValueError("Please provide a valid PyTorch model")

        # Saving the model
        self.__model = model
