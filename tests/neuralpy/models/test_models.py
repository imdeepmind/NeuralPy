from neuralpy.models import Model, Sequential
from neuralpy.layers import Dense
from neuralpy.loss_functions import MSELoss
from neuralpy.optimizer import Adam
from neuralpy import device

import pytest
import numpy as np

np.random.seed(1969)

X_train = np.random.rand(100, 1) * 10
y_train = X_train + 5 * np.random.rand(100, 1)

X_validation = np.random.rand(100, 1) * 10
y_validation = X_validation + 5 * np.random.rand(100, 1)

X_test = np.random.rand(10, 1) * 10
y_test = X_test + 5 * np.random.rand(10, 1)

model = Sequential()
model.add(Dense(n_nodes=1, n_inputs=1))

model.build()

pytorch_model = model.get_model()

def test_model():
    with pytest.raises(ValueError) as ex:
        x = Model(force_cpu="test")
    
    with pytest.raises(ValueError) as ex:
        x = Model(training_device="test")
    
    with pytest.raises(ValueError) as ex:
        x = Model(random_state="test")
    
    training_device = device("cpu")
    
    x1 = Model(force_cpu=False, training_device=training_device, random_state=1969)
    x2 = Model(force_cpu=False, training_device=None, random_state=1969)
    x3 = Model(force_cpu=True, training_device=None, random_state=1969)
    
def test_model_fit_method():
    model = Model()
    model.set_model(pytorch_model)
    model.compile(optimizer=Adam(), loss_function=MSELoss())

    model.fit(train_data=(X_train, y_train), validation_data=(
    X_validation, y_validation), epochs=1, batch_size=32)

    with pytest.raises(ValueError) as ex:
        model.fit(train_data=(X_train, y_train), validation_data=(X_validation, y_validation), epochs=1, batch_size=1024)
    
    with pytest.raises(ValueError) as ex:
        model.fit(train_data=(X_train, y_train[:-1]), validation_data=(X_validation, y_validation), epochs=1, batch_size=1024)    