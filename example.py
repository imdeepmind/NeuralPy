from neuralpy.models import Sequential
from neuralpy.layers import Dense
from neuralpy.activation_functions import sigmoid, relu, softmax
from neuralpy.loss_functions import binary_cross_entropy, mean_squared_error

X = [[1,2,3], [4,5,6]]
y = [1, 2]

model = Sequential()

model.add(Dense(n_nodes=256, n_inputs=784, bias=True, name="Input Layer"))
model.add(Dense(n_nodes=10, n_inputs=None, bias=True, name="Output Layer"))

model.build()
model.summary()









# model.forward(X)

# model.add(Dense(n_inputs=3, n_nodes=5, activation=relu))
# model.add(Dense(n_inputs=5, n_nodes=1, activation=sigmoid))

# model.compile(optimizer=None, loss_function=mean_squared_error)

# output = model.predict(X)

# loss = model.evaluate(X, y)

# model.save("ignore/model.pickle")

# print(model.summary())
# print(output)
# print("Loss of the model", loss)

