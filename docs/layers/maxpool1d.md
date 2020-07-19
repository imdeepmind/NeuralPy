# Conv1D

```python
neuralpy.layers.MaxPool1D(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, name=None)
```

MaxPool1d Applies a 1D max pooling over an input.

To learn more about MaxPool1D layers, please check PyTorch [documentation](https://pytorch.org/docs/stable/nn.html#maxpool1d)

## Supported Arguments:

  - `kernel_size`: (Integer | Tuple) Kernel size of the layer
  - `stride`: (Integer | Tuple) Stride for the conv.
  - `padding`: (Integer | Tuple) Padding for the conv layer.
  - `dilation`: (Integer | Tuple) Controls the spacing between the kernel elements
  - `return_indices`: (Boolean) If True, will return the max indices along with the outputs.
  - `ceil_mode`: (Boolean) When True, will use ceil instead of floor to compute the output shape
  - `name`: (String) Name of the layer, if not provided then automatically calculates a unique name for the layer

### Example Code

```python
from neuralpy.models import Sequential
from neuralpy.layers import Conv1D, MaxPool1D

# Making the model
model = Sequential()
model.add(Conv1D(filters=8, kernel_size=3, input_shape=(1, 28), stride=1, name="first cnn"))
model.add(MaxPool1D(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False, name="Pool Layer"))
```