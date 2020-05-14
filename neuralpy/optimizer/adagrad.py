from torch.optim import Adagrad as _Adagrad

class Adagrad:
    def __init__(self, learning_rate=0.001, learning_rate_decay=0, eps=1e-08, weight_decay=0):
        if not 0.0 <= learning_rate:
            raise ValueError("Invalid learning_rate")
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilion")
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value")

        self.__learning_rate = learning_rate
        self.__learning_rate_decay = learning_rate_decay
        self.__eps = eps
        self.__weight_decay = weight_decay

    def get_optimizer(self):
        return {
            'optimizer': _Adagrad,
            'keyword_arguments': {
                'lr': self.__learning_rate,
                'lr_decay': self.__learning_rate_decay,
                'weight_decay': self.__weight_decay
            }
        }
