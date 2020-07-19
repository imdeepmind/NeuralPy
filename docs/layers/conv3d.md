# Conv1D

```python
neuralpy.layers.Conv3D(filters, kernel_size, input_shape=None, stride=1, padding=0, dilation=1, groups=1, bias=True, name=None)
```

Applies a 3D convolution over an input signal composed of several input planes.

To learn more about Conv3D layers, please check PyTorch [documentation](https://pytorch.org/docs/stable/nn.html#conv3d)

## Supported Arguments:

  - `filters`: (Integer) Size of the filter
  - `kernel_size`: (Integer | Tuple) Kernel size of the layer
  - `input_shape`: (Tuple) A tuple with the shape in following format (input_channel, X, Y, Z). No need of this argument layers except the initial layer.
  - `stride`: (Integer | Tuple) Stride for the conv.
  - `padding`: (Integer | Tuple) Padding for the conv layer.
  - `dilation`: (Integer | Tuple) Controls the spacing between the kernel elements
  - `groups`: (Integer) Controls the connections between inputs and outputs.
  - `bias`: (Boolean) If true then uses the bias, Defaults to `true`
  - `name`: (String) Name of the layer, if not provided then automatically calculates a unique name for the layer

### Example Code

```python
from neuralpy.models import Sequential
from neuralpy.layers import Conv3D

# Making the model
model = Sequential()
model.add(Conv3D(filters=8, kernel_size=3, input_shape=(1, 28, 28, 28), stride=1, name="first cnn"))
model.add(Conv3D(filters=16, kernel_size=3, stride=1, name="second cnn"))
```