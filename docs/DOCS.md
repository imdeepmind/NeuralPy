
# Introduction
NeuralPy is a Keras like, deep learning library that works on top of PyTorch written purely in Python. It is simple, easy to use library that is cross-compatible with PyTorch models and suitable for all kinds of machine learning experiments, learning, research, etc.

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

# Test data
X_test = np.random.rand(100, 1) * 10
y_test = X_test + 5 * np.random.rand(100, 1)
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

#### Building the model
Once the model archeitectre is ready, we can build the model. This build methods a PyTorch model inernally.

```python
# Building the model
model.build()
```

#### Compiling the model
Once the model architecture is ready, we can compile with the optimizer and loss function. NeuralPy models provide an easy compile method for that. We just need to pass the optimizer and loss function in the compile method. It also accepts a metrics parameter, for this case, we don't need it.

```python
# Compiling the model
model.compile(optimizer=Adam(), loss_function=MSELoss())
```

#### Training the model
To train we have the fit method. We need to pass the training and validation data, along with some other parameters to the fit method.

```python
model.fit(train_data=(X_train, y_train), test_data=(X_validation, y_validation), epochs=300, batch_size=4)
```

#### Predicting Results
Main purpose of model is predict. In NeuralPy models, there are two methods for prediction,  `.predict()` and `.predict_classes()`. Here for this linear regression problem, we'll use the `.predict()` method.

```python
# Predicting
preds = model.predict(X=X_test, batch_size=4)
```

#### Evaluating the models
After training, one importent step is to evaluate the model on the test dataset. To do that, we have, in NeuralPy, we've a `.evaluate()`

---

# Models
```python
neuralpy.models
```
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

#### `.add()` method: 
In a Sequential model, the `.add()` method is responsible for adding a new layer to the model. It accepts a NeuralPy layer class as an argument and builds a model, and based on that. The .add() method can be called as many times as needed. There is no limitation on that, assuming you have enough computation power to handle it.

##### Supported Arguments
- `layer`: (NeuralPy layer classes) Adds a layer into the model
##### Example Codes
```python
from neuralpy.models import Sequential
from neuralpy.layers import Dense
from neuralpy.activation_functions import ReLU
from neuralpy import device

# Setting a training device
training_device = device('cpu')

# Initializing the Sequential models
model = Sequential(force_cpu=False, training_device=training_device, random_state=1969)

# Adding layers to the model
model.add(Dense(n_nodes=3, n_inputs=5, bias=True))
model.add(ReLU())
model.add(Dense(n_nodes=3, n_inputs=3, bias=True))
```
#### `.build()` method:

In a Sequential model, the `.build()` method is responsible for building the PyTorch model from the NeuralPy model.

After finishing the architecture of the model, the model needed to be built before training.

##### Supported Arguments:
- There is no argument for this model

#### Exaample Code
```python
from neuralpy.models import Sequential
...
# Rest of the imports
...

model = Sequential()
...
# Model Architecture
...

# Calling .build to build the model
model.build()
```

#### `.compile()` mehod:
In the Sequential model, the compile method is responsible for attaching a loss function and optimizer with the model and this method needs to be called before training.

> The `.compile()` method internally calls the `.build()`, so there is no need to call 	`.build()`.

	 
##### Supported Arguments:
- `optimizer`: (NeuralPy Optimizer class) Adds an optimizer to the model
- `loss_function`: (NeuralPy Loss Function class) Adds a loss function to the model
- `metrics`: ([String]) Metrics that will be evaluated by the model. Currently only supports `accuracy`. 

#### Exaample Code
```python
from neuralpy.models import Sequential
from neuralpy.optimizer import Adam
from neuralpy.loss_functions import MSELoss
...
# Rest of the imports
...

model = Sequential()
...
# Model Architecture
...

# Calling .compile to build the model 
# and attach a optimizer and loss function with the model
model.compile(optimizer=Adam(), loss_function=MSELoss(),metrics=["accuracy"])
```

#### `.fit()` Method	
The `.fit()` method is used for training the NeuralPy model. 

##### Supported Arguments
- `train_data`: (Tuple(NumPy Array, NumPy Array)) Pass the training data as a tuple like `(X, y)` where `X` is training data and `y` is the labels for the training the model.
- `test_data`:(Tuple(NumPy Array, NumPy Array)) Pass the validation data as a tuple like `(X, y)` where `X` is test data and `y` is the labels for the validating the model.
- `epochs=10`: (Integer) Number of epochs
- `batch_size=32`: (Integer) Batch size for training.

##### Example Code
```python
```python
from neuralpy.models import Sequential
...
# Rest of the code
...

# Training the model
model.fit(train_data, test_data, epochs=10, batch_size=32)
```
#### `.predict()` Method
The `.predict()`method is used for predicting using the trained mode.

##### Supported Arguments
- `X`: (NumPy Array) Data to be predicted
- `batch_size=None`: (Integer) Batch size for predicting. If not provided, then the entire data is predicted once.

##### Example Code
```python
from neuralpy.models import Sequential
...
# Rest of the code
...

# Predicting using the trained model
y_pred = model.predict(X, batch_size=32)
```

#### `.predict_class()` Method
The `.predict_clas()` method is used for predicting classes using the trained model. This method works only if `accuracy` is passed in the `metrics` parameter on the `.compile()` method.

##### Supported Arguments
- `X`: (NumPy Array) Data to be predicted
- `batch_size=None`: (Integer) Batch size for predicting. If not provided, then the entire data is predicted once.

##### Example Code
```python
from neuralpy.models import Sequential
...
# Rest of the code
...

# Predicting the labels using the trained model
y_pred = model.predict_clas(X, batch_size=32)
```


#### `.evaluate()` Method
The `.evaluate()` method is used for evaluating models using the test dataset.

##### Supported Arguments
- `X`: (NumPy Array) Data to be predicted
- `y`: (NumPy Array) Original labels of `X`
- `batch_size=None`: (Integer) Batch size for predicting. If not provided, then the entire data is predicted once.

##### Example Code
```python
from neuralpy.models import Sequential
...
# Rest of the code
...

# Evaluating the labels using the trained model
results = model.evaluate(X, batch_size=32)
```




#### `.summary()` Method
The `.summary()` method is getting a summary of the model

##### Supported Arguments
- None

##### Example Code
```python
from neuralpy.models import Sequential
...
# Rest of the code
...

# Detailed summary of the model
print(model.summary())
```




#### `.get_model()` Method
The `.get_model()` method is used for getting the PyTorch model from the NeuralPy model. After extracting the model, the model can be treated just like a regular PyTorch model. 

##### Supported Arguments
- None

##### Example Code
```python
from neuralpy.models import Sequential
...
# Rest of the code
...

# Extracting the PyTorch model
pytorch_model = model.get_model()
```


#### `.set_model()` Method
The `.set_model()` method is used for converting a PyTorch model to a NeuralPy model. After this conversion, the model can be trained using NeuralPy optimizer and loss_functions.

##### Supported Arguments
- `model`: (PyTorch model) A valid class based on Sequential PyTorch model.

##### Example Code
```python
from neuralpy.models import Sequential
...
# Rest of the code
...

# Conveting the PyTorch model to NeuralPy model
model.set_model(pytorch_model)
```

---


# Layers
```python
neuralpy.layers
```
Layers are the building blocks of a Neural Network. A complete Neural Network model consists of several layers. A Layer is a function that receives a tensor as output, computes something out of it, and finally outputs a tensor.

NeuralPy currently supports only one type of Layer and that is the Dense layer. 

## Dense Layer
```python
neuralpy.layers.Dense(n_nodes, n_inputs=None, bias=True, name=None)
```
A Dense is a normal densely connected Neural Network. It performs a linear transformation of the input.

To learn more about Dense layers, please check [pytorch documentation](https://pytorch.org/docs/stable/nn.html?highlight=linear#torch.nn.Linear) for it.

### Supported Arguments
- `n_nodes`: (Integer) Size of the output sample
- `n_inputs=None`: (Integer) Size of the input sample, no need for this argument layers except the initial layer
- `bias=True`: (Boolean) If true then uses the bias
- `name=None`: (String) Name of the layer, if not provided then automatically calculates a unique name for the layer

### Example Code
```python
from neuralpy.models import Sequential
from neuralpy.layers import Dense

# Making the model
model = Sequential()
model.add(Dense(n_nodes=256, n_inputs=28, bias=True, name="Input Layer"))
model.add(Dense(n_nodes=512, bias=True, name="Hidden Layer 1"))
model.add(Dense(n_nodes=10, bias=True, name="Output Layer"))
```

---

# Activation Functions
```python
neuralpy.activation_functions
```
Activation Functions are simple functions that tell a neuron to fire or not, the purpose is to introduce non-linearity to Neural Network layers.

NeuralPy supports various activation functions that are widely used for building complex Neural Network models. 

## ReLU
```python
neuralpy.activation_functions.ReLU(name=None)
```
ReLU applies a rectified linear unit function to the input tensors.

For more information, check [this](https://pytorch.org/docs/stable/nn.html#relu) page

###  Supported Arguments
- `name=None`: (String) Name of the activation function layer, if not provided then automatically calculates a unique name for the layer

### Example Code
```python
from neuralpy.models import Sequential
from neuralpy.layers import Dense
from neuralpy.activation_functions import ReLU

# Initializing the Sequential models
model = Sequential()

# Adding layers to the model
model.add(Dense(n_nodes=3, n_inputs=5, bias=True))
model.add(ReLU())
model.add(Dense(n_nodes=3, bias=True))
```
## LeakyReLU
```python
neuralpy.activation_functions.LeakyReLU(negative_slope=0.01, name=None)
```
LeakyReLU is a modified ReLU activation function with some improvements. LeakyReLU solves the problem of "dead ReLU", by introducing a new parameter called the negative slope. 

In traditional ReLU, if the input is negative, then the output is 0. But for LeakyReLU, the output is not zero. This feature special behavior of LeakyReLU solves the problem of "dead ReLU" and helps in learning.

For more information, check [this](https://pytorch.org/docs/stable/nn.html#leakyrelu) page

###  Supported Arguments
- `negative_slope=0.01`: (Integer) A negative slope for the LeakyReLU
- `name=None`: (String) Name of the activation function layer, if not provided then automatically calculates a unique name for the layer

### Example Code
```python
from neuralpy.models import Sequential
from neuralpy.layers import Dense
from neuralpy.activation_functions import LeakyReLU

# Initializing the Sequential models
model = Sequential()

# Adding layers to the model
model.add(Dense(n_nodes=3, n_inputs=5, bias=True))
model.add(LeakyReLU())
model.add(Dense(n_nodes=3, bias=True))
```
## Softmax
```python
neuralpy.activation_functions.Softmax(dim=None, name=None)
```
Applies the Softmax function to the input Tensor rescaling input to the range [0,1].

For more information, check [this](https://pytorch.org/docs/stable/nn.html#softmax) page

###  Supported Arguments
- `dim=None`: (Integer) A dimension along which Softmax will be computed (so every slice along dim will sum to 1).
- `name=None`: (String) Name of the activation function layer, if not provided then automatically calculates a unique name for the layer.

### Example Code
```python
from neuralpy.models import Sequential
from neuralpy.layers import Dense
from neuralpy.activation_functions import LeakyReLU, Softmax

# Initializing the Sequential models
model = Sequential()

# Adding layers to the model
model.add(Dense(n_nodes=256, n_inputs=28, bias=True))
model.add(LeakyReLU())

model.add(Dense(n_nodes=512, bias=True))
model.add(LeakyReLU())

model.add(Dense(n_nodes=10, bias=True))
model.add(Softmax())
```

## Sigmoid
```python
neuralpy.activation_functions.Sigmoid(name=None)
```
Applies a element-wise Sigmoid or Logistic function to the input tensor.

For more information, check [this](https://pytorch.org/docs/stable/nn.html#sigmoid) page

###  Supported Arguments
- `name=None`: (String) Name of the activation function layer, if not provided then automatically calculates a unique name for the layer.

### Example Code
```python
from neuralpy.models import Sequential
from neuralpy.layers import Dense
from neuralpy.activation_functions import LeakyReLU, Sigmoid

# Initializing the Sequential models
model = Sequential()

# Adding layers to the model
model.add(Dense(n_nodes=256, n_inputs=28, bias=True))
model.add(LeakyReLU())

model.add(Dense(n_nodes=512, bias=True))
model.add(LeakyReLU())

model.add(Dense(n_nodes=1, bias=True))
model.add(Sigmoid())
```

## Tanh
```python
neuralpy.activation_functions.Tanh(name=None)
```
Applies a element-wise Tanh function to the input tensor.

For more information, check [this](https://pytorch.org/docs/stable/nn.html#tanh) page

###  Supported Arguments
- `name=None`: (String) Name of the activation function layer, if not provided then automatically calculates a unique name for the layer.

### Example Code
```python
from neuralpy.models import Sequential
from neuralpy.layers import Dense
from neuralpy.activation_functions import LeakyReLU, Tanh

# Initializing the Sequential models
model = Sequential()

# Adding layers to the model
model.add(Dense(n_nodes=256, n_inputs=28, bias=True))
model.add(Tanh())

model.add(Dense(n_nodes=128, bias=True))
model.add(Tanh())
```

---

# Regulariziers
```python
neuralpy.regulariziers
```
Regularization is a technique that helps a model to generalize well on datasets, and it helps to overcome the problem of Overfitting.

## Dropout
```python
neuralpy.regulariziers.Dropout(p=0.5, name=None)
```
Applies the Dropout layer to the input tensor.

The Dropout layer randomly sets input units to 0 with a frequency of rate of `p` at each step during training time. It helps prevent overfitting.

For more information, check [this](https://pytorch.org/docs/stable/nn.html#dropout) page

###  Supported Arguments
- `p=0.5`: (Float) Probability of an element to be zeroed. The value should be between 0.0 and 1.0.
- `name=None`: (String) Name of the layer, if not provided then automatically calculates a unique name for the layer

### Example Code
```python
from neuralpy.models import Sequential
from neuralpy.layers import Dense
from neuralpy.activation_functions import LeakyReLU, Sigmoid
from neuralpy.regulariziers import Dropout

# Initializing the Sequential models
model = Sequential()

# Adding layers to the model
model.add(Dense(n_nodes=3, n_inputs=5, bias=True))
model.add(LeakyReLU())
model.add(Dropout())

model.add(Dense(n_nodes=20, bias=True))
model.add(LeakyReLU())
model.add(Dropout())

model.add(Dense(n_nodes=1, bias=True))
model.add(Sigmoid())
```

---


# Loss Functions
```python
neuralpy.loss_functions
```
Loss Functions are functions that calculate the error rate of a model. The optimizer optimizes the model based on these Loss Functions.

NeuralPy currently supports 3 types of Loss Functions, BCELoss, CrossEntropyLoss, and MeanSquaredLoss.

## BCE Loss
```python
neuralpy.loss_functions.BCELoss(weight=None, reduction='mean', pos_weight=None)
```
Applies a BCE Loss function to the model. 

> BCE Loss automatically applies a Sigmoid Layer at the end of the model, so there is no need to add a Sigmoid layer.

For more information, check [this](https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss) page.

### Supported Arguments
- `weight=None` : (Numpy Array | List) Manual rescaling of classes
- `reduction='mean'` : (String) Specifies the reduction that is to be applied to the output.
- `post_weight=None` : (Numpy Array | List)  A weight of positive examples

### Code Example
```python
from neuralpy.models import Sequential
from neuralpy.optimizer import Adam
from neuralpy.loss_functions import BCELoss
...
# Rest of the imports
...

model = Sequential()
...
# Rest of the architecture
...

model.compile(optimizer=Adam(), loss_function=BCELoss(weight=None, reduction='mean', pos_weight=None))
```

## Cross Entropy Loss
```python
neuralpy.loss_functions.CrossEntropyLoss(weight=None, ignore_index=-100 reduction='mean')
```
Applies a Cross-Entropy Loss function to the model. 

> Cross-Entropy Loss automatically applies a Softmax Layer at the end of the model, so there is no need to add a Softmax layer.

For more information, check [this](https://pytorch.org/docs/stable/nn.html#crossentropyloss) page.

### Supported Arguments
- `weight=None` : (Numpy Array | List) Manual rescaling of classes
- `ignore_index=-100` : (Integer)  Specifies a target value that is ignored and does not contribute to the input gradient.
- `reduction='mean'` : (String) Specifies the reduction that is to be applied to the output.

### Code Example
```python
import numpy as np
from neuralpy.models import Sequential
from neuralpy.optimizer import Adam
from neuralpy.loss_functions import BCELoss
...
# Rest of the imports
...

model = Sequential()
...
# Rest of the architecture
...
# Weight of different classes, here 3 is the number of classes
weight = np.ones([3])

model.compile(optimizer=Adam(), loss_function=CrossEntropyLoss(weight=weight, reduction='mean', pos_weight=None))
```

## MSE Loss
```python
neuralpy.loss_functions.MSELoss(reduction='mean')
```
Applies a Mean Squared Error loss function to the model. 

For more information, check [this](https://pytorch.org/docs/stable/nn.html#mseloss) page.

### Supported Arguments
- `reduction='mean'` : (String) Specifies the reduction that is to be applied to the output.

### Code Example
```python
from neuralpy.models import Sequential
from neuralpy.optimizer import Adam
from neuralpy.loss_functions import MSELoss
...
# Rest of the imports
...

model = Sequential()
...
# Rest of the architecture
...
# Compiling the model
model.compile(optimizer=Adam(), loss_function=MSELoss(reduction='mean'))
```



---

# Optimizers 
```python neuralpy.optimizer ``` 

 Optimizers are one of the most important parts of Machine Learning. The `neuralpy.optimizer` package implements different types of optimizers that can be used to optimizer a neuralpy model.

Currently there 4 types of the optimizer, that neuralpy supports, these are Adam, Adagrad, RMSProp, and SGD.

## SGD
```python
neuralpy.optimizer.SGD(learning_rate=0.001, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False)
```
Applies an SGD (Stochastic Gradient Descent) with momentum.

### Supported Arguments
- `learning_rate=0.001`: (Float) Learning Rate for the optimizer
- `momentum=0` : (Float) Momentum for the optimizer
- `dampening=0` : (Float) Dampening of momentum
- `weight_decay=0` : (Float) Weight decay for the optimizer
- `nesterov=False` : (Bool) Enables Nesterov momentum

### Example Code
```python
from neuralpy.models import Sequential
from neuralpy.optimizer import SGD
from neuralpy.loss_functions import MSELoss
...
# Rest of the imports
...

model =  Sequential()
...
# Rest of the model
...

# Compiling the model
model.compile(
	optimizer=SGD(learning_rate=0.001, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False), 
	loss_function=MSELoss())
```

## Adam
```python
neuralpy.optimizer.Adam(learning_rate=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False)
```
Implements Adam optimizer.

### Supported Arguments
- `learning_rate=0.001`: (Float) Learning Rate for the optimizer
- `betas=(0.9,0.999)` : (Tuple[Float, Float]) coefficients used for computing running averages of gradient and its square
- `eps=0` : (Float) Term added to the denominator to improve numerical stability
- `weight_decay=0` : (Float) Weight decay for the optimizer
- `amsgrad=False` : (Bool) if true, then uses AMSGrad various of the optimizer

### Example Code
```python
from neuralpy.models import Sequential
from neuralpy.optimizer import Adam
from neuralpy.loss_functions import MSELoss
...
# Rest of the imports
...

model =  Sequential()
...
# Rest of the model
...

# Compiling the model
model.compile(
	optimizer=Adam(learning_rate=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False), 
	loss_function=MSELoss())
```

## Adagrad
```python
neuralpy.optimizer.Adagrad(learning_rate=0.001, learning_rate_decay=0.0, eps=1e-08, weight_decay=0.0)
```
Implements Adagrad optimizer.

### Supported Arguments
- `learning_rate=0.001`: (Float) Learning Rate for the optimizer
- `learning_rate_decay=(0.9,0.999)` : (Float) Learningn Rate decay
- `eps=0` : (Float) Term added to the denominator to improve numerical stability
- `weight_decay=0` : (Float) Weight decay for the optimizer

### Example Code
```python
from neuralpy.models import Sequential
from neuralpy.optimizer import Adagrad
from neuralpy.loss_functions import MSELoss
...
# Rest of the imports
...

model =  Sequential()
...
# Rest of the model
...

# Compiling the model
model.compile(
	optimizer=Adagrad(learning_rate=0.001, learning_rate_decay=0.0, eps=1e-08, weight_decay=0.0), 
	loss_function=MSELoss())
```

## RMSProp
```python
neuralpy.optimizer.RMSProp(learning_rate=0.001, alpha=0.99, eps=1e-08, weight_decay=0.0,momentum=0.0, centered=False)
```
Implements RMSProp optimizer.

### Supported Arguments
- `learning_rate=0.001`: (Float) Learning Rate for the optimizer
- `alpha=(0.9,0.999)` : (Float) Learningn Rate decay
- `eps=0` : (Float) Term added to the denominator to improve numerical stability
- `weight_decay=0` : (Float) Weight decay for the optimizer
- `momentum=0` : (Float) Momentum for the optimizer
- `centered=False` : (Bool) if `True`, compute the centered RMSProp, the gradient is normalized by an estimation of its variance

### Example Code
```python
from neuralpy.models import Sequential
from neuralpy.optimizer import RMSProp
from neuralpy.loss_functions import MSELoss
...
# Rest of the imports
...

model =  Sequential()
...
# Rest of the model
...

# Compiling the model
model.compile(
	optimizer=RMSProp(learning_rate=0.001, alpha=0.99, eps=1e-08, weight_decay=0.0,momentum=0.0, centered=False), 
	loss_function=MSELoss())
```

---


# Advanced Topics

NeuralPy is limited because there are limited types of Layers, Loss Functions, Optimizers, Regularizers, etc, in NeuralPy.

But there are times when we might need a layer, or an optimizer, or a loss function for a model that is not implimented on NeuralPy.

In NeuralPy, anyone can build a custom Layer, Optimizer, Loss Function, Regularizer, etc.

## Building a Custom Layer

NeuralPy is based on PyTorch, so first we need to build PyTorch valid layer, or use some existing PyTorch layer. To get a list of Layers implemented by PyTorch, check [this](https://pytorch.org/docs/stable/nn.html) page.

> The PyTorch layer needed to be a class-based layer, functional PyTorch layers are not supported by NeuralPy.

> Also if you want to build your own custom PyTorch layer, then please check [this](https://hackernoon.com/how-to-build-your-own-pytorch-neural-network-layer-from-scratch-2x6136th) medium article.

#### So now let's start coding

First import the PyTorch layer class that you want to use in NeuralPy. If you are using a custom PyTorch layer, then import that.

In the example below, We'll use the [Flatten](https://pytorch.org/docs/stable/nn.html#flatten) layer, This layer already is implemented by PyTorch. So importing it from the `nn` module.

```python
from torch.nn import Flatten as _Flatten
...
# Rest of the imports
...
```

After that, create a class with two public methods `get_input_dim` and `get_layer`. Along with that create an `__init__` method also.

```python
class Flatten:
    def __init__(self):
       pass

    def get_input_dim(self, prev_input_dim):
       pass

    def get_layer(self):
       pass
```

Here the `__init__` method is for setting up the layer. Pass all the parameters that you need for the layer, like input share, output shape, etc.

The `get_input_dim` method is used for calculating input shape based on the output shape of the previous layer. If your layer does not have an input shape, then just return `None`.

The `get_layer` method is the most important layer and it returns a dictionary with all the details that NeuralPy Model class needs for building the model.

```python
class Flatten:
    def __init__(self, start_dim=1, end_dim=-1, name):
	# Validate the parameters

	# Checking the name field, this is an optional field,
	# if not provided generates a unique name for the layer
        if name is not None and not (isinstance(name, str) and name):
            raise ValueError("Please provide a valid name")

	    self.__start_dim = start_dim
	    self.__end_dim = end_dim
	    self.__name = name

    def get_input_dim(self, prev_input_dim):

        # As there is no input shape, returning None
        return None

    def get_layer(self):
        return {
             'n_inputs': None,
             'n_nodes': None,
             'name': self.__name,
             'type': 'Flatten',
             'layer': _Flatten,
             'keyword_arguments': {
                 'start_dim': self.__start_dim,
                 'end_dim': self.__end_dim
              }
        }
```
If you check the PyTorch docs, then Flatten accepts two parameters, `start_dim` and `end_dim`. So in the `__init__` method, I've added these two parameters, along with the `name` parameter. NeuralPy needs a name for every layer, if there is no name provided, then auto generates a layer name.

Flatten does not have an input shape parameter, `get_input_dim` method just returns `None`.

Finally, the `get_layer` method returns a dictionary with several fields. Here is the detail of all the fields.

- `n_inputs`: Pass the input shape of the layer, in the next layer, you'll get this field as `prev_input_dim` parameter in the `get_input_dim`.

- `n_nodes`: Is the output dim of the layer

- `type`: Type of the layer, just pass the layer name in a string. This is used to auto-generate layer name

- `name`: Pass the name parameter

- `keyword_arguments`: It contains a dictionary of all the parameters that the PyTorch layer or your custom layer accepts. If there is no parameter for the layer, send set it as None. For our `Flatten` layer, we need to pass the `start_dim` and `end_dim`.  

> We can create Activation Functions, Regularizers using the same above method also.

Check the following links for some more examples:
- [Dense layer](https://github.com/imdeepmind/NeuralPy/blob/master/neuralpy/layers/dense.py)
- [ReLU Activation Function](https://github.com/imdeepmind/NeuralPy/blob/master/neuralpy/activation_functions/relu.py)
- [Dropout](https://github.com/imdeepmind/NeuralPy/blob/master/neuralpy/regulariziers/dropout.py)

## Building a custom Optimizer
Steps for making a custom Optimizer is mostly similar to making a custom Layer. So first we need to import a PyTorch optimizer or use a custom PyTorch compatible optimizer.

Create a class with one public method `get_optimizer` and an `__init__` method.

```python
from torch.optim import SGD as _SGD

class SGD:
	def __init__(self):
		pass
	
	def get_optimizer(self):
		pass
```

Then in the `__init__` method pass all the parameters that is needed for the optimizer.

In the `get_optimizer` method returns a dictionary with all the details that is needed by the NeuralPy Models.

```python
from torch.optim import SGD as _SGD

class SGD:
    def __init__(self, learning_rate=0.001, momentum=0.0,
                 dampening=0.0, weight_decay=0.0, nesterov=False):

        # Validation the input fields
        if not isinstance(learning_rate, float) or learning_rate < 0.0:
            raise ValueError("Invalid learning_rate")

        if not isinstance(momentum, float) or momentum < 0.0:
            raise ValueError("Invalid momentum value")

        if not isinstance(weight_decay, float) or weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value")

        if not isinstance(dampening, float):
            raise ValueError("Invalid dampening parameter")

        if not isinstance(nesterov, bool):
            raise ValueError("Invalid nesterov parameter")

        self.__learning_rate = learning_rate
        self.__momentum = momentum
        self.__dampening = dampening
        self.__weight_decay = weight_decay
        self.__nesterov = nesterov

    def get_optimizer(self):
        # Returning the optimizer data
        return {
            'optimizer': _SGD,
            'keyword_arguments': {
                'lr': self.__learning_rate,
                'momentum': self.__momentum,
                'dampening': self.__dampening,
                'weight_decay': self.__weight_decay,
                'nesterov': self.__nesterov
            }
        }
```

Here the `get_optimizer` is similar to the `get_layer` method. Just returns a dictionary with all the data. The `optimizer` in the dictionary is the main optimizer and the `keyword_arguments` contains all the arguments for the PyTorch SGD optimizer.



## Building a custom Loss Function
Steps for making a custom Loss Function is also mosty similar to making a custom Layer. So first we need to import a PyTorch loss function or use a custom PyTorch compatible loss function.

Create a class with one public method `get_loss_function` and an `__init__` method.

```python
from torch.nn import MSELoss as _MSELoss

class SGD:
	def __init__(self):
		pass
	
	def get_loss_function(self):
		pass
```

Then in the `__init__` method pass all the parameters that are needed for the optimizer.

In the `get_loss_function` method returns a dictionary with all the details that are needed by the NeuralPy Models.

```python
from torch.nn import MSELoss as _MSELoss

class MSELoss:
    def __init__(self, reduction='mean'):
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError("Invalid reduction")
           
        self.__reduction = reduction

    def get_loss_function(self):
        return {
            'loss_function': _MSELoss,
            'keyword_arguments': {
                'reduction': self.__reduction
            }
        }
```

Here the `get_loss_function` is similar to the `get_layer` method. Just returns a dictionary with all the data. The `loss_function`  field in the dictionary is the main loss function and the `keyword_arguments` contains all the arguments for the PyTorch MSE Loss Function.





---
<!--stackedit_data:
eyJoaXN0b3J5IjpbNDUzNTc0ODAyLC0xNjkxMTM1NzU3LDE3OD
A1MDg3MjNdfQ==
-->