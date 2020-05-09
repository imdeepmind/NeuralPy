from neuralpy.models import Sequential
from neuralpy.layers import Dense
from neuralpy.activation_functions import ReLU, Sigmoid, Softmax
from neuralpy.optimizer import Adam
from neuralpy.loss_functions import MSELoss, CrossEntropyLoss

import torch

X = torch.randn(1000, 1) * 10
y = X + 5 * torch.randn(1000, 1)

model = Sequential()

model.add(Dense(n_nodes=1, n_inputs=1, bias=True, name="Input Layer"))
# model.add(())

# model.add(Dense(n_nodes=256, n_inputs=None, bias=True, name="Hidden Layer"))
# model.add(ReLU())

# model.add(Dense(n_nodes=10, n_inputs=None, bias=True, name="Output Layer"))
# model.add(Softmax())

model.compile(optimizer=Adam(), loss_function=MSELoss())

model.summary()

history = model.fit(X, y, epochs=500, batch_size=32)









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

