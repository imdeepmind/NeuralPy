"""TrainLogger class"""

class TrainLogger:
    def __init__(self, path):
        self.__path = path
    
    def callback(self, epochs, epoch, loss_function_parameters, optimizer_parameters, traning_progress):
        print(traning_progress)