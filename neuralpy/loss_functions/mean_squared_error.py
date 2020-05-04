import numpy as np

def mean_squared_error(y, y_pred):
	return (np.sum(y - y.pred)**2) / len(y)