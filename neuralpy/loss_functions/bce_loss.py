from torch.nn import BCEWithLogitsLoss as _BCEWithLogitsLoss

class BCELoss:
	def __init__(self, weight=None, reduction='mean'):
		self.__weight = weight
		self.__reduction = reduction

	def get_loss_function(self):
		return {
			'loss_function': _BCEWithLogitsLoss,
			'keyword_arguments': {
				'weight': self.__weight,
				'reduction': self.__reduction
			}
		}