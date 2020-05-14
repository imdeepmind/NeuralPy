from torch.nn import BCELoss as _BCELoss

class BCELoss:
	def __init__(self, weight=None, reduction='mean'):
		self.__weight = weight
		self.__reduction = reduction

	def get_loss_function(self):
		return {
			'loss_function': _BCELoss,
			'keyword_arguments': {
				'weight': self.__weight,
				'reduction': self.__reduction
			}
		}