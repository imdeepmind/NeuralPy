# Introduction
NeuralPy is a Keras like, machine learning library that works on top of PyTorch written purely in Python. It is simple, easy to use library, cross-compatible with PyTorch models, suitable for all kinds of machine learning experiments, learning, research, etc.

## PyTorch in NeuralPy
PyTorch is an open-source machine learning framework that accelerates the path from research prototyping to production deployment developed by Facebook runs on both CPU and GPU.

Following are some highlights of PyTorch:
- Production Ready
- Distributed Training
- Robust Ecosystem
- Cloud Support

NeuralPy is a high-level library that works on top of PyTorch. It provides easy to use, powerful and interactive APIs that can be used to build state-of-the-art models. As it works on top of PyTorch, NerualPy supports both CPU and GPU and can perform numerical operations very efficiently.

Here are some highlights of NeuralPy:
- Provides an easy interface that is suitable for fast prototyping, learning, and research
- Works on top of PyTorch
- Can run on both CPU and GPU
- Cross-Compatible with PyTorch models
    
## Get Started with NeuralPy
Let’s get started with NeuralPy. The first step is to install NeuralPy, but before installing NerualPy, we need to install some additional dependencies.

### Prerequisites 
NeuralPy runs on Python 3, so first install Python 3. Please check the python documentation for installation instructions. After that install PyTorch and Numpy. For instructions, please check their official documentation and installation guides.

### Installation
To install NeuralPy run the following command on terminal.
```bash
pip install neuralpy-torch
```
If you have multiple versions of it, then you might need to use pip3.

```bash
pip3 install neuralpy-torch
//or
python3 -m pip install neuralpy-torch
```

### First Model
Here, in this example, we’ll create a simple linear regression model. The model accepts X (independent variable) and predicts the y(dependent variable). The model basically leans the relation between X and y. 

#### Dataset for the model
Here we’ll create some synthetic data for our model, we’ll use numpy to generate random numbers.

```python
# Importing dependencies
import numpy as np

# Random seed for numpy
np.random.seed(1969)

# Generating the data
# Training data
X_train = np.random.rand(100, 1) * 10
y_train = X_train + 5 * np.random.rand(100, 1)

# Validation data
X_validation = np.random.rand(100, 1) * 10
y_validation = X_validation + 5 * np.random.rand(100, 1)
```

#### Importing dependencies from NeuralPy
Let’s import the dependencies from NeuralPy. 

Here we’ll use the Sequential model. Sequential is basically a linear stack of layers. Dense as a layer, Adam as an optimizer and MSELoss for calculating the loss.

```python
from neuralpy.models import Sequential
from neuralpy.layers import Dense
from neuralpy.optimizer import Adam
from neuralpy.loss_functions import MSELoss
```

#### Making the model
Sequential provides an easy .add() method to add layers in the sequential model. We’ll use it to build the model.

```python
# Making the model
model = Sequential()

# Adding the layer
model.add(Dense(n_nodes=1, n_inputs=1))
```

#### Compiling the model
Once the model architecture is ready, we can compile with the optimizer and loss function. The Sequential class provides an easy compile method for that. We just need to pass the optimizer and loss function in the compile method.

```python
# Compiling the model
model.compile(optimizer=Adam(), loss_function=MSELoss())
```

#### Training the model
To train we have the fit method. We need to pass the training and validation data, along with some other parameters to the fit method.

```python
model.fit(train_data=(X_train, y_train), test_data=(X_validation, y_validation), epochs=300, batch_size=4)
```

# Models
Models are one of the most important API supported in NeuralPy. Models are used to create different architecture. In NeuralPy, currently Sequential is the only type of model that is supported.

## Sequential
```python
neuralpy.models.Sequential(force_cpu=False, training_device=None, random_state=None)
```
Sequential is a linear stack of layers with single input and output layer. It is one of the simplest types of models. In Sequential models, each layer has a single input and output tensor.

### Supported Arguments
- `force_cpu=False`: (Boolean) If True, then uses CPU even if CUDA is available
- `training_device=None`: (NeuralPy device class) Device that will be used for training predictions
- `random_state`: (Integer) Random state for the device

### Supported Methods
#### `.add() method`: 
In a Sequential model, the .add() method is responsible for adding a new layer to the model. It accepts a NeuralPy layer class as an argument and builds a model, and based on that. The .add() method can be called as many times as needed. There is no limitation on that, assuming you have enough computation power to handle it.

##### Supported Arguments
- `layer`: (NeuralPy layer classes) Adds a layer into the model
##### Example Codes
```python
from neuralpy.models import Sequential
from neuralpy.layers import Dense
from neuralpy import device

# Setting a training device
training_device = device('cpu')

# Initializing the Sequential models
model = Sequential(force_cpu=False, training_device=training_device, random_state=1969)

# Adding layers to the model
model.add(Dense(n_nodes=3, n_inputs=5, bias=True))
model.add(Dropout())
model.add(ReLU())
model.add(Dense(n_nodes=3, n_inputs=3, bias=True))
```