from torch.nn import MSELoss as _MSELoss

class MSELoss:
	def __init__(self, reduction='mean'):
		self.__reduction = reduction

	def get_loss_function(self):
		return {
			'loss_function': _MSELoss,
			'keyword_arguments': {
				'size_average': None, # Deprecated
				'reduce': None, # Deprecated
				'reduction': self.__reduction
			}
		}