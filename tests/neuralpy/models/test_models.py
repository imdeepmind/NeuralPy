from neuralpy.models import Model, Sequential
from neuralpy.layers import Dense
from neuralpy.loss_functions import MSELoss
from neuralpy.optimizer import Adam
from neuralpy.callbacks import TrainLogger
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

def train_generator():
    for i in range(40):
        X_train = np.random.rand(40, 1) * 10
        y_train = X_train + 5 * np.random.rand(40, 1)

        yield X_train, y_train

def predict_generator():
    for i in range(40):
        X_train = np.random.rand(40, 1) * 10
        y_train = X_train + 5 * np.random.rand(40, 1)

        yield X_train

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

def test_models_compile_method():
    model = Model()
    model.set_model(pytorch_model)
    model.compile(optimizer=Adam(), loss_function=MSELoss())

    with pytest.raises(ValueError) as ex:
        model = Model()
        model.set_model(pytorch_model)

        model.compile(optimizer=Adam(), loss_function=MSELoss(), metrics=["test"])
    
    with pytest.raises(ValueError) as ex:
        model = Model()
        model.set_model(pytorch_model)

        model.compile(optimizer=Adam(), loss_function=MSELoss(), metrics="test")
    
def test_model_fit_method():
    model = Model()
    model.set_model(pytorch_model)
    model.compile(optimizer=Adam(), loss_function=MSELoss())
    logger = TrainLogger("ignore/")

    model.fit(train_data=(X_train, y_train), validation_data=(X_validation, y_validation), epochs=1, batch_size=32)
    
    model.fit(train_data=(X_train, y_train), validation_data=(X_validation, y_validation), epochs=1, batch_size=32, callbacks=[logger])

    train_gen = train_generator()
    validation_gen = train_generator()

    model.fit(train_data=train_gen, validation_data=validation_gen, epochs=1, batch_size=4, steps_per_epoch=5, validation_steps=5)

    model.fit(train_data=train_gen, validation_data=validation_gen, epochs=1, batch_size=4, steps_per_epoch=5, validation_steps=5, callbacks=[logger])

    with pytest.raises(ValueError) as ex:
        model.fit(train_data=(X_train, y_train), validation_data=(X_validation, y_validation), epochs=1, batch_size=1024)
    
    with pytest.raises(ValueError) as ex:
        model.fit(train_data=(X_train, y_train[:-1]), validation_data=(X_validation, y_validation), epochs=-20, batch_size=1024)    
    
    with pytest.raises(ValueError) as ex:
        model.fit(train_data=(X_train, y_train[:-1]), validation_data=(X_validation, y_validation), epochs=1, batch_size=-10)    

    with pytest.raises(ValueError) as ex:
        model.fit(train_data=(X_train, y_train[:-1]), validation_data=(X_validation, y_validation), epochs=1, batch_size=32, callbacks="test") 

    with pytest.raises(ValueError) as ex:
        train_gen = train_generator()
        validation_gen = train_generator()

        model.fit(train_data=train_gen, validation_data=validation_gen, epochs=1, batch_size=32, steps_per_epoch=-123, validation_steps=5)
    
    with pytest.raises(ValueError) as ex:
        train_gen = train_generator()
        validation_gen = train_generator()

        model.fit(train_data=train_gen, validation_data=validation_gen, epochs=1, batch_size=32, steps_per_epoch="test", validation_steps=5)
    
    with pytest.raises(ValueError) as ex:
        train_gen = train_generator()
        validation_gen = train_generator()

        model.fit(train_data=train_gen, validation_data=validation_gen, epochs=1, batch_size=32, steps_per_epoch=5, validation_steps=-23)
    
    with pytest.raises(ValueError) as ex:
        train_gen = train_generator()
        validation_gen = train_generator()

        model.fit(train_data=train_gen, validation_data=validation_gen, epochs=1, batch_size=32, steps_per_epoch=5, validation_steps="asd")
        