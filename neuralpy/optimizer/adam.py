from torch.nn import Adam as _Adam

class Adam:
	def __init__(self, learning_rate=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):
		if not 0.0 <= learning_rate:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.__learning_rate = learning_rate
        self.__betas = betas
        self.__eps = eps
        self.__weight_decay = weight_decay
        self.__amsgrad = amsgrad

	def get_optimizer():
		return {
			'optimizer': _Adam,
			'keyword_arguments': {
				'lr': self.__learning_rate,
				'betas': self.__betas,
				'eps': self.__eps,
				'weight_decay': self.__weight_decay,
				'amsgrad': self.__amsgrad
			}
		}