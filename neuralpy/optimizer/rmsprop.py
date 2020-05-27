from torch.optim import RMSprop as _RMSprop


class RMSprop:
    def __init__(self, learning_rate=0.001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False):
        if not 0.0 <= learning_rate:
            raise ValueError("Invalid learning_rate")
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilion")
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value")

        self.__learning_rate = learning_rate
        self.__alpha = alpha
        self.__eps = eps
        self.__weight_decay = weight_decay
        self.__momentum = momentum
        self.__centered = centered

    def get_optimizer(self):
        return {
            'optimizer': _RMSprop,
            'keyword_arguments': {
                'lr': self.__learning_rate,
                'alpha': self.__alpha,
                'eps': self.__eps,
                'weight_decay': self.__weight_decay,
                'momentum': self.__momentum,
                'centered': self.__centered
            }
        }
