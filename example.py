from neuralpy.models import Sequential
from neuralpy.layers import Dense
from neuralpy.activation_functions import sigmoid, relu
from neuralpy.loss_functions import binary_cross_entropy, mean_squared_error

input = [[1,2,3], [4,5,6]]

model = Sequential()

model.add(Dense(n_inputs=3, n_nodes=5, activation=relu))
model.add(Dense(n_inputs=5, n_nodes=1, activation=sigmoid))

model.compile(optimizer=None, loss_function=mean_squared_error)

output = model.predict(input)

model.save("ignore/model.pickle")

print(model.summary())
print(output)

