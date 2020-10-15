"""TrainLogger class"""
import datetime
import os

# pylint: disable=too-few-public-methods


class TrainLogger:
    """
        TrainLogger is a callback for NeuralPy models fit method. It is used for creating
        training logs with different parameters

        Supported Arguments:
            path: (String) path where the log files will be stored

    """

    def __init__(self, path):
        """
            __init__ method for TrainLogger class

            Supported Arguments:
                path: (String) path where the log files will be stored
        """
        if not path or not isinstance(path, str):
            raise ValueError("Please provide a valid path")

        self.__headers = []
        self.__rows = []

        filename = (
            str(datetime.datetime.now()) +
            ".log").replace(" ", "_").replace("-", "_").replace(":", "_")

        self.__path = os.path.join(path, filename)

        if not os.path.exists(path):
            os.makedirs(path)

    def __generate_log_file(self):
        text = ",".join(self.__headers) + "\n"

        for row in self.__rows:
            text += ",".join(row) + "\n"

        with open(self.__path, "w") as file:
            file.write(text)

    # pylint: disable=too-many-arguments,unused-argument
    def callback(self, epochs, epoch, loss_function_parameters, optimizer_parameters,
                 traning_progress, model):
        """
            The callback method is called from the model class once it completes an epoch

            Supported Arguments:
                epochs: (Integer) Total number of epochs
                epoch: (integer) Current epoch
                loss_function_parameters: (Dictionary) All parameters of the loss function
                optimizer_parameters: (Dictionary) All parameters of the optimizer
                traning_progress: (Dictionary) Training progress of the current epoch
        """
        headers = ['epochs', 'epoch']
        row = [str(epochs), str(epoch)]
        if loss_function_parameters:
            for k in loss_function_parameters:
                headers.append(k)
                row.append(str(loss_function_parameters[k]))

        if optimizer_parameters:
            for k in optimizer_parameters:
                headers.append(k)
                row.append(str(optimizer_parameters[k]))

        if traning_progress:
            for k in traning_progress:
                headers.append(k)
                row.append(str(traning_progress[k]))

        self.__headers = headers
        self.__rows.append(row)

        self.__generate_log_file()
