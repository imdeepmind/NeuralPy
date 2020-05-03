from neuralpy.layers import Dense
from neuralpy.models import Sequential


input = [[1,2,3], [4,5,6]]

model = Sequential()

model.add(Dense(3, 5))
model.add(Dense(5, 1))


print(model.summary())

output = model.predict(input)

print(output)