from torch.optim import SGD as _SGD


class SGD:
    def __init__(self, learning_rate=0.001, momentum=0,
                 dampening=0, weight_decay=0, nesterov=False):
        if learning_rate < 0.0:
            raise ValueError("Invalid learning_rate")
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value")

        self.__learning_rate = learning_rate
        self.__momentum = momentum
        self.__dampening = dampening
        self.__weight_decay = weight_decay
        self.__nesterov = nesterov

    def get_optimizer(self):
        return {
            'optimizer': _SGD,
            'keyword_arguments': {
                'lr': self.__learning_rate,
                'momentum': self.__momentum,
                'dampening': self.__dampening,
                'weight_decay': self.__weight_decay,
                'nesterov': self.__nesterov
            }
        }
