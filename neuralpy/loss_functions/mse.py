from torch.nn import MSELoss as _MSELoss

# pylint: disable=too-few-public-methods
class MSELoss:
    def __init__(self, reduction='mean'):
        self.__reduction = reduction

        if not isinstance(reduction, str) and reduction not in ["none", "mean", "sum"]:
            raise ValueError("Invalid reduction")

    def get_loss_function(self):
        return {
            'loss_function': _MSELoss,
            'keyword_arguments': {
                'reduction': self.__reduction
            }
        }
