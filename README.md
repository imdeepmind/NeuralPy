
<p align="center">
 <img src="https://user-images.githubusercontent.com/34741145/81591141-99752900-93d9-11ea-9ef6-cc2c68daaa19.png" alt="Logo of NeuralPy" />
 <br />
 A Keras like deep learning library works on top of PyTorch
</p>

![NeuralPy Build Check](https://github.com/imdeepmind/NeuralPy/workflows/NeuralPy%20Build%20Check/badge.svg)
![Maitained](https://img.shields.io/badge/Maitained%3F-Yes-brightgreen)
![PyPI - Downloads](https://img.shields.io/pypi/dw/neuralpy-torch?style=flat)
![PyPI](https://img.shields.io/pypi/v/neuralpy-torch?style=flat)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/imdeepmind/NeuralPy?style=flat)
![GitHub issues](https://img.shields.io/github/issues/imdeepmind/NeuralPy?style=flat)
![GitHub](https://img.shields.io/github/license/imdeepmind/NeuralPy?style=flat)

## Table of contents:
- [Table of contents:](#table-of-contents)
- [Introduction](#introduction)
- [PyTorch](#pytorch)
- [Install](#install)
- [Dependencies](#dependencies)
- [Get Started](#get-started)
  - [Importing the dependencies](#importing-the-dependencies)
  - [Making some random data](#making-some-random-data)
  - [Making the model](#making-the-model)
  - [Training the model](#training-the-model)
  - [Predicting using the trained model](#predicting-using-the-trained-model)
- [Documentation](#documentation)
- [Examples](#examples)
- [Blogs and Tutorials](#blogs-and-tutorials)
- [Support](#support)
- [Contributing](#contributing)
- [License](#license)

## Introduction
NeuralPy is a High-Level [Keras](https://keras.io/) like deep learning library that works on top of [PyTorch](https://pytorch.org) written in pure Python. NeuralPy can be used to develop state-of-the-art deep learning models in a few lines of code. It provides a Keras like simple yet powerful interface to build and train models. 

Here are some highlights of NeuralPy
 - Provides an easy interface that is suitable for fast prototyping, learning, and research
 - Can run on both CPU and GPU
 - Works on top of PyTorch
 - Cross-Compatible with PyTorch models

## PyTorch
PyTorch is an open-source machine learning framework that accelerates the path from research prototyping to production deployment developed by Facebook runs on both CPU and GPU.

According to Wikipedia, 
> PyTorch is an open-source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing, primarily developed by Facebook's AI Research lab (FAIR). It is free and open-source software released under the Modified BSD license.

NeuralPy is a high-level library that works on top of PyTorch. As it works on top of PyTorch, NerualPy supports both CPU and GPU and can perform numerical operations very efficiently.

If you want to learn more about PyTorch, then please check the [PyTorch documentation](https://pytorch.org/).

## Install
To install NeuralPy, open terminal window type the following command:
```
pip install neuralpy-torch
```
If you have multiple versions of it, then you might need to use pip3.
```
pip3 install neuralpy-torch
//or
python3 -m pip install neuralpy-torch
```
> NeuralPy requires Pytorch and Numpy, first install those

Check the documentation for Installation related information

## Dependencies
The only dependencies of NeuralPy are Pytorch (used as backend) and Numpy.

## Get Started
Let's create a linear regression model in 100 seconds.

### Importing the dependencies
```python
import numpy as np

from neuralpy.models import Sequential
from neuralpy.layers.linear import Dense
from neuralpy.optimizer import Adam
from neuralpy.loss_functions import MSELoss
```

### Making some random data
```python
# Random seed for numpy
np.random.seed(1969)

# Generating the data
X_train = np.random.rand(100, 1) * 10
y_train = X_train + 5 *np.random.rand(100, 1)

X_validation = np.random.rand(100, 1) * 10
y_validation = X_validation + 5 * np.random.rand(100, 1)

X_test = np.random.rand(10, 1) * 10
y_test = X_test + 5 * np.random.rand(10, 1)
```

### Making the model
```python
# Making the model
model = Sequential()
model.add(Dense(n_nodes=1, n_inputs=1, bias=True, name="Input Layer"))

# Building the model
model.build()

# Compiling the model
model.compile(optimizer=Adam(), loss_function=MSELoss())

# Printing model summary
model.summary()
```

### Training the model
```python
model.fit(train_data=(X_train, y_train), validation_data=(X_validation, y_validation), epochs=300, batch_size=4)
```

### Predicting using the trained model
```python
model.predict(predict_data=X_test, batch_size=4)
```

## Documentation
The documentation for NeuralPy is available at [https://www.neuralpy.xyz/](https://www.neuralpy.xyz/)

## Examples  
Several example projects in NeuralPy are available at [https://github.com/imdeepmind/NeuralPy-Examples](https://github.com/imdeepmind/NeuralPy-Examples). Please check the above link.

## Blogs and Tutorials
Following are some links to official blogs and tutorials:
  - [Introduction to NeuralPy: A Keras like deep learning library works on top of PyTorch](https://medium.com/@imdeepmind/introduction-to-neuralpy-a-keras-like-deep-learning-library-works-on-top-of-pytorch-3bbf1b887561)

## Support
If you are facing any issues using NeuralPy, then please raise an issue on GitHub or contact with me. 

Alternatively, you can join the official NeuralPy discord server. Click [here](https://discord.gg/6aTTwbW) to join.

## Contributing
Feel free to contribute to this project. If you need some help to get started, then reach me or open a GitHub issue. Check the [CONTRIBUTING.MD](https://github.com/imdeepmind/NeuralPy/blob/master/CONTRIBUTING.md) file for more information and guidelines.

## License
[MIT](https://github.com/imdeepmind/NeuralPy/blob/master/LICENSE)

