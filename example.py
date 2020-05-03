from neuralpy.layers import Dense
from neuralpy.models import Sequential


input = [[1,2,3], [4,5,6]]

model = Sequential()

model.add(Dense, {})


print(model.layers)
# dense_layer = Dense(3, 1)
# dense_layer.forward(input)

# print("Output of the network", dense_layer.output)
# print("Weights of the network", dense_layer.get_weights())