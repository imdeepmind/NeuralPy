"""NeuralPy Model Class"""

import torch

from .model_helper import (is_valid_optimizer,
                           is_valid_loss_function,
                           build_optimizer_from_dict,
                           build_loss_function_from_dict,
                           build_history_object,
                           calculate_accuracy,
                           print_training_progress,
                           print_validation_progress)

class Model:
    """
        NeuralPy Model Class
    """
    def __init__(self, force_cpu, training_device, random_state):
        self.__model = None
        self.__metrics = ["loss"]
        self.__loss_function = None
        self.__optimizer = None

        # Checking the force_cpu parameter
        if not isinstance(force_cpu, bool):
            raise ValueError(
                f"You have provided an invalid value for the parameter force_cpu")

        # Checking the training_device parameter and comparing it with pytorch device class
        # pylint: disable=no-member
        if training_device and not isinstance(training_device, torch.device):
            raise ValueError("Please provide a valid neuralpy device class")

        # Validating random state
        if random_state and not isinstance(random_state, int):
            raise ValueError("Please provide a valid random state")

        # if force_cpu then using CPU
        # if device provided, then using it
        # else auto detecting the device, if cuda available then using it (default option)

        # there is a issue pylint, because of that, disabling the no-member check
        # for more info, have the look at the link below
        # https://github.com/pytorch/pytorch/issues/701

        if training_device:
            self.__device = training_device
        elif force_cpu:
            # pylint: disable=no-member
            self.__device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                # pylint: disable=no-member
                self.__device = torch.device("cuda:0")
            else:
                # pylint: disable=no-member
                self.__device = torch.device("cpu")

        # Setting random state if given
        if random_state:
            torch.manual_seed(random_state)

    # pylint: disable=invalid-name
    def __predict(self, X, batch_size=None):
        """
            Method for predicting

            Supported Arguments:
                X: (Numpy Array) Data
                batch_size=None: (Integer) Batch size for predicting
        """
        # Calling model.eval as we are evaluating the model only
        self.__model.eval()

        # Initializing an empty list to store the predictions
        # pylint: disable=not-callable,no-member
        predictions = torch.Tensor().to(self.__device)

        # Converting the input X to PyTorch Tensor
        X = torch.tensor(X)

        if batch_size:
            # If batch_size is there then checking the length
            # and comparing it with the length of input
            if X.shape[0] < batch_size:
                # Batch size can not be greater that sample size
                raise ValueError(
                    "Batch size is greater than total number of samples")

            # Predicting, so no grad
            with torch.no_grad():
                # Splitting the data into batches
                for i in range(0, len(X), batch_size):
                    # Generating the batch from X
                    batch_x = X[i:i+batch_size].float().to(self.__device)

                    # Feeding the batch into the model for predictions
                    outputs = self.__model(batch_x)

                    # Appending the data into the predictions tensor
                    # pylint: disable=not-callable,no-member
                    predictions = torch.cat((predictions, outputs))
        else:
            # Predicting, so no grad
            with torch.no_grad():
                # Feeding the full data into the model for predictions tensor
                outputs = self.__model(X.float().to(self.__device))

                # saving the outputs in the predictions
                predictions = outputs

        # returning predictions tensor
        return predictions

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
        if not is_valid_optimizer(optimizer):
            raise ValueError("Please provide a value neuralpy optimizer")

        # Checking the loss_function using the method is_valid_loss_function
        if not is_valid_loss_function(loss_function):
            raise ValueError("Please provide a value neuralpy loss function")

        if metrics and not isinstance(metrics, list):
            raise ValueError("Please provide a valid metrics")

        # Setting metrics
        if metrics:
            self.__metrics = ["loss"] + metrics

        # Storing the loss function and optimizer for future use
        self.__optimizer = build_optimizer_from_dict(optimizer,
                                                     self.__model.parameters())
        self.__loss_function = build_loss_function_from_dict(loss_function)

    def fit(self, train_data, test_data, epochs=10, batch_size=32):
        """
            The `.fit()` method is used for training the NeuralPy model.

            Supported Arguments
                train_data: (Tuple(NumPy Array, NumPy Array)) Pass the training data
                    as a tuple like (X, y) where X is training data and y is the
                    labels for the training the model.
                test_data:(Tuple(NumPy Array, NumPy Array)) Pass the validation data
                    as a tuple like (X, y) where X is test data and y is the labels
                    for the validating the model.
                epochs=10: (Integer) Number of epochs
                batch_size=32: (Integer) Batch size for training.


        """
        # Extracting the train and test data from the tuples
        x_train, y_train = train_data
        x_test, y_test = test_data

        # If batch_size is there then checking the
        # length and comparing it with the length of training data
        if x_train.shape[0] < batch_size:
            # Batch size can not be greater that train data size
            raise ValueError(
                "Batch size is greater than total number of training samples")

        # If batch_size is there then checking the length and
        # comparing it with the length of training data
        if x_test.shape[0] < batch_size:
            # Batch size can not be greater that test data size
            raise ValueError(
                "Batch size is greater than total number of testing samples")

        # Checking the length of input and output
        if x_train.shape[0] != y_train.shape[0]:
            # length of X and y should be same
            raise ValueError(
                "Length of training Input data and training output data should be same")

        # Checking the length of input and output
        if x_test.shape[0] != y_test.shape[0]:
            # length of X and y should be same
            raise ValueError(
                "Length of testing Input data and testing output data should be same")

        # Converting the data into PyTorch tensor
        # pylint: disable=not-callable,no-member
        x_train = torch.tensor(x_train)
        y_train = torch.tensor(y_train)

        x_test = torch.tensor(x_test)
        y_test = torch.tensor(y_test)

        # Building the history object
        history = build_history_object(self.__metrics)

        # Running the epochs
        for epoch in range(epochs):
            # Initializing the loss and accuracy with 0
            training_loss_score = 0
            validation_loss_score = 0

            correct_training = 0
            correct_val = 0

            # Training model :)
            self.__model.train()

            # Splitting the data into batches
            for i in range(0, len(x_train), batch_size):
                # Making the batches
                batch_x = x_train[i:i+batch_size].float()
                if "accuracy" in self.__metrics:
                    batch_y = y_train[i:i+batch_size]
                else:
                    batch_y = y_train[i:i+batch_size].float()

                # Moving the batches to device
                batch_x, batch_y = batch_x.to(
                    self.__device), batch_y.to(self.__device)

                # Zero grad
                self.__model.zero_grad()

                # Feeding the data into the model
                outputs = self.__model(batch_x)

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
                if "accuracy" in self.__metrics:
                    corrects = calculate_accuracy(batch_y, outputs)

                    correct_training += corrects

                    history["batchwise"]["training_accuracy"].append(
                        corrects/batch_size*100)

                    print_training_progress(epoch, epochs, i, batch_size, len(
                        x_train), train_loss.item(), corrects)
                else:
                    print_training_progress(
                        epoch, epochs, i, batch_size, len(x_train), train_loss.item())

            # Evaluating model
            self.__model.eval()

            # no grad, no training
            with torch.no_grad():
                # Splitting the data into batches
                for i in range(0, len(x_test), batch_size):
                    # Making the batches
                    batch_x = x_train[i:i+batch_size].float()
                    if "accuracy" in self.__metrics:
                        batch_y = y_train[i:i+batch_size]
                    else:
                        batch_y = y_train[i:i+batch_size].float()

                    # Moving the batches to device
                    batch_x, batch_y = batch_x.to(
                        self.__device), batch_y.to(self.__device)

                    # Feeding the data into the model
                    outputs = self.__model(batch_x)

                    # Calculating the loss
                    validation_loss = self.__loss_function(outputs, batch_y)

                    # Storing the loss val, batchwise data
                    validation_loss_score += validation_loss.item()
                    history["batchwise"]["validation_loss"].append(
                        validation_loss.item())

                    # Calculating accuracy
                    # Checking if accuracy is there in metrics
                    if "accuracy" in self.__metrics:
                        corrects = corrects = calculate_accuracy(
                            batch_y, outputs)

                        correct_val += corrects

                        history["batchwise"]["validation_accuracy"].append(
                            corrects/batch_size*100)

            # Calculating the mean val loss score for all batches
            validation_loss_score /= batch_size

            # Added the epochwise value to the history dictionary
            history["epochwise"]["training_loss"].append(training_loss_score)
            history["epochwise"]["validation_loss"].append(
                validation_loss_score)

            # Checking if accuracy is there in metrics
            if "accuracy" in self.__metrics:
                # Adding data into history dictionary
                history["epochwise"]["training_accuracy"].append(
                    correct_training/len(x_train)*100)
                history["epochwise"]["training_accuracy"].append(
                    correct_val/len(x_test)*100)

                # Printing a friendly message to the console
                print_validation_progress(
                    validation_loss_score, len(x_test), correct_val)
            else:
                # Printing a friendly message to the console
                print_validation_progress(
                    validation_loss_score, len(x_test))

        # Returning history
        return history

    def predict(self, X, batch_size=None):
        """
            The .predict()method is used for predicting using the trained mode.

            Supported Arguments
                X: (NumPy Array) Data to be predicted
                batch_size=None: (Integer) Batch size for predicting.
                If not provided, then the entire data is predicted once.
        """
        # Calling the __predict method to get the predicts
        predictions = self.__predict(X, batch_size)

        # Returning an numpy array of predictions
        return predictions.numpy()

    def predict_classes(self, X, batch_size=None):
        """
            The .predict_clas()method is used for predicting classes using the trained mode.
            This method works only if accuracy is passed in the metrics parameter on the
            .compile()method.

            Supported Arguments
                X: (NumPy Array) Data to be predicted
                batch_size=None: (Integer) Batch size for predicting.
                If not provided, then the entire data is predicted once.
        """
        # Checking if the model is for classification
        if self.__metrics and "accuracy" in self.__metrics:
            # Calling the __predict method to get the predicts
            predictions = self.__predict(X, batch_size)

            # Detecting the classes
            predictions = predictions.argmax(dim=1, keepdim=True)

            return predictions.numpy()
        else:
            raise ValueError(
                "Cannot predict classes as this is not a classification problem")

    def evaluate(self, X, y, batch_size=None):
        """
            The .evaluate()method is used for evaluating models using the test dataset.

            Supported Arguments
                X: (NumPy Array) Data to be predicted
                y: (NumPy Array) Original labels of X
                batch_size=None: (Integer) Batch size for predicting.
                    If not provided, then the entire data is predicted once.
        """
        # If batch_size is there then checking the length and comparing
        # it with the length of training data
        if batch_size and X.shape[0] < batch_size:
            # Batch size can not be greater that train data size
            raise ValueError(
                "Batch size is greater than total number of training samples")

        # Checking the length of input and output
        if X.shape[0] != y.shape[0]:
            # length of X and y should be same
            raise ValueError(
                "Length of training Input data and training output data should be same")

        # Calling the __predict method to get the predicts
        predictions = self.__predict(X, batch_size)

        # Converting to tensor
        # pylint: disable=not-callable,no-member
        if self.__metrics and "accuracy" in self.__metrics:
            y_tensor = torch.tensor(y).to(self.__device)
        else:
            y_tensor = torch.tensor(y).float().to(self.__device)

        # Calculating the loss
        loss = self.__loss_function(predictions, y_tensor)

        # if metrics has accuracy, then calculating accuracy
        if self.__metrics and "accuracy" in self.__metrics:
            # Calculating no of corrects
            corrects = calculate_accuracy(y_tensor, predictions)

            # Calculating accuracy
            accuracy = corrects / len(X) * 100

            # Returning loss and accuracy
            return {
                'loss': loss.item(),
                'accuracy': accuracy
            }

        # Returning loss
        return {
            'loss': loss.item()
        }

    def summary(self):
        """
            The .summary() method is getting a summary of the model.

            Supported Arguments
                None
        """
        # Printing the model summary using PyTorch model
        if self.__model:
            # Printing models summary
            print(self.__model)

            # Calculating total number of params
            print("Total Number of Parameters: ", sum(p.numel()
                                                      for p in self.__model.parameters()))

            # Calculating total number of trainable params
            print("Total Number of Trainable Parameters: ", sum(p.numel()
                                                                for p in self.__model.parameters()
                                                                if p.requires_grad))
        else:
            raise Exception("You need to build the model first")

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

        # Transferring the model to device
        self.__model.to(self.__device)

        # Printing a message with the device name
        print("The model is running on", self.__device)
