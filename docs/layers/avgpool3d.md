# Conv1D

```python
neuralpy.layers.AvgPool3D(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, name=None)
```

AvgPool3D Applies a 3D avg pooling over an input.

To learn more about AvgPool3D layers, please check PyTorch [documentation](https://pytorch.org/docs/stable/nn.html#avgpool3d)

## Supported Arguments:

  - `kernel_size`: (Integer | Tuple) Kernel size of the layer
  - `stride`: (Integer | Tuple) Stride for the conv.
  - `padding`: (Integer | Tuple) Padding for the conv layer.
  - `ceil_mode`: (Boolean) When True, will use ceil instead of floor to compute the output shape
  - `count_include_pad`: (Boolean) When True, will include the zero-padding in the averaging calculation
  - `name`: (String) Name of the layer, if not provided then automatically calculates a unique name for the layer

### Example Code

```python
from neuralpy.models import Sequential
from neuralpy.layers import Conv3D, AvgPool3D

# Making the model
model = Sequential()
model.add(Conv3D(filters=8, kernel_size=3, input_shape=(1, 28, 28, 28), stride=1, name="first cnn"))
model.add(AvgPool3D(kernel_size=3, stride=3, padding=0, ceil_mode=False, count_include_pad=True, name="Pool Layer"))
```