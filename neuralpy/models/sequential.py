class Sequential:
	layers = []

	def __init__(self):
		pass

	def add(self, layer, configuration):
		self.layers.append({
			'layer':layer, 
			'configuration':configuration
		})



