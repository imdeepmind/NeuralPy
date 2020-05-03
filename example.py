from neuralpy.models import Sequential
from neuralpy.layers import Dense
from neuralpy.activation_functions import sigmoid


input = [[1,2,3], [4,5,6]]

model = Sequential()

model.add(Dense(3, 5))
model.add(Dense(5, 1, activation=sigmoid))


print(model.summary())

output = model.predict(input)

print(output)