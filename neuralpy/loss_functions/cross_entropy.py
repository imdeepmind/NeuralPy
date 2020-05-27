from torch.nn import CrossEntropyLoss as _CrossEntrypyLoss


class CrossEntropyLoss:
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        self.__weight = weight
        self.__ignore_index = ignore_index
        self.__reduction = reduction

    def get_loss_function(self):
        return {
            'loss_function': _CrossEntrypyLoss,
            'keyword_arguments': {
                'weight': self.__weight,
                'ignore_index': self.__ignore_index,
                'reduction': self.__reduction
            }
        }
