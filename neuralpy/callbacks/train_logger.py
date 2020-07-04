"""TrainLogger class"""
import datetime
import os


class TrainLogger:
    def __init__(self, path):
        self.__headers = []
        self.__rows = []

        filename = (str(datetime.datetime.now()) + ".log").replace(" ",
                                                                   "_").replace("-", "_").replace(":",
                                                                                                  "_")

        self.__path = os.path.join(path, filename)

    def __generate_log_file(self):
        text = ",".join(self.__headers) + "\n"

        for row in self.__rows:
            text += ",".join(row) + "\n"

            
        with open(self.__path, "w") as f:
            f.write(text)

    def callback(self, epochs, epoch, loss_function_parameters, optimizer_parameters, traning_progress):
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
