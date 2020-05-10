from neuralpy.models import Sequential
from neuralpy.layers import Dense
from neuralpy.activation_functions import ReLU, Sigmoid, Softmax
from neuralpy.optimizer import Adam
from neuralpy.loss_functions import MSELoss, CrossEntropyLoss

from numpy import random

X_train = random.rand(1000, 1) * 10
y_train = X_train + 5 * random.rand(1000, 1)

X_test = random.rand(100, 1) * 10
y_test = X_test + 5 * random.rand(100, 1)

model = Sequential()

model.add(Dense(n_nodes=1, n_inputs=1, bias=True, name="Input Layer"))
# model.add(())

# model.add(Dense(n_nodes=256, n_inputs=None, bias=True, name="Hidden Layer"))
# model.add(ReLU())

# model.add(Dense(n_nodes=10, n_inputs=None, bias=True, name="Output Layer"))
# model.add(Softmax())

model.compile(optimizer=Adam(), loss_function=MSELoss())

model.summary()

history = model.fit(train_data=(X_train, y_train), test_data=(X_test, y_test), epochs=100, batch_size=32)

# print(history["epochwise"])









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

