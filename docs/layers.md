# Layers

```python
neuralpy.layers
```
Layers are the building blocks of a Neural Network. A complete Neural Network model consists of several layers. A Layer is a function that receives a tensor as output, computes something out of it, and finally outputs another transformed tensor.

NeuralPy currently supports only one type of Layer and that is the Dense layer. 

## Linear Layers

### Dense Layer

```python
neuralpy.layers.Dense(n_nodes, n_inputs=None, bias=True, name=None)
```
A Dense is a normal densely connected Neural Network. It performs a linear transformation of the input.

To learn more about Dense layers, please check [pytorch documentation](https://pytorch.org/docs/stable/nn.html?highlight=linear#torch.nn.Linear) for it.

#### Supported Arguments

- `n_nodes`: (Integer) Size of the output sample
- `n_inputs=None`: (Integer) Size of the input sample, no need for this argument layers except the initial layer
- `bias=True`: (Boolean) If true then uses the bias
- `name=None`: (String) Name of the layer, if not provided then automatically calculates a unique name for the layer

#### Example Code

```python
from neuralpy.models import Sequential
from neuralpy.layers import Dense

# Making the model
model = Sequential()
model.add(Dense(n_nodes=256, n_inputs=28, bias=True, name="Input Layer"))
model.add(Dense(n_nodes=512, bias=True, name="Hidden Layer 1"))
model.add(Dense(n_nodes=10, bias=True, name="Output Layer"))
```

### Bilinear Layer

```python
neuralpy.layers.Bilinear(n_nodes, n1_features=None, n2_features=None, bias=True, name=None)
```

Applies a bilinear transformation to the incoming data. 

A bilinear layer is a function of two inputs x and y that is linear in each input separately. Simple bilinear functions on vectors are the dot product or the element-wise product.

To learn more about Bilinear layers, please check [pytorch documentation](https://pytorch.org/docs/stable/nn.html#bilinear) for it.

#### Supported Arguments

- `n_nodes`: (Integer) Size of the output sample
- `n1_features=None`: (Integer) Size of the input sample 1, no need for this argument layers except the initial layer.
- `n2_features`: (Integer) Size of the input sample 2.
- `bias=True`: (Boolean) If true then uses the bias
- `name=None`: (String) Name of the layer, if not provided then automatically calculates a unique name for the layer

#### Example Code

```python
from neuralpy.models import Sequential
from neuralpy.layers import Bilinear

# Making the model
model = Sequential()
model.add(Bilinear(n_nodes=256, n1_features=20, n2_features=20, bias=True, name="Input Layer"))
```

## Convolution Layers

Convolution layers are the building blocks of Convolutional Neural Networks. A Convolution layer is a type of specialized layer that is 

### Conv1D layer

### Conv2D Layer

### Conv3D Layer

## Pooling Layers

### MaxPool1D

### MaxPool2D

### MaxPool3D

### AvgPool1D

### AvgPool2D

### AvgPool3D

## Other Layers

Other utility layers used in a variety of models.

### Flatten Layer

```python
neuralpy.layers.Flatten(start_dim=1, end_dim=-1, name=None)
```

Flattens the output from a layer. Usually used after Conv layers.

To learn more about Dense layers, please check [PyTorch documentation](https://pytorch.org/docs/stable/nn.html?highlight=flatten#torch.nn.Flatten)

Supported Arguments:

- `start_dim=1`: (Integer) Start dim for flatten
- `end_dim=-1`: (Integer) End dim for flatten
- `name=None`: (String) Name of the layer, if not provided then automatically calculates a unique name for the layer

