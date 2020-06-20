"""MSE Loss Function"""

from torch.nn import MSELoss as _MSELoss

# pylint: disable=too-few-public-methods

class MSELoss:
    """
        Applies a Mean Squared Error loss function to the model.

        Supported Arguments
            reduction='mean' : (String) Specifies the reduction that
                is to be applied to the output.
    """

    def __init__(self, reduction='mean'):
        """
            __init__ method for MSELoss

            Supported Arguments
                reduction='mean' : (String) Specifies the reduction that
                    is to be applied to the output.
        """
        self.__reduction = reduction

        if reduction not in ["none", "mean", "sum"]:
            raise ValueError("Invalid reduction")

    def get_loss_function(self):
        """
            Returns the details of the loss function

            There is no need to call this method as this is used by the
            Sequential model to build the model
        """
        return {
            'loss_function': _MSELoss,
            'keyword_arguments': {
                'reduction': self.__reduction
            }
        }
