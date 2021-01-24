from neuralpy.models.model_helper import (
    generate_layer_name,
    build_layer_from_dict,
    build_optimizer_from_dict,
    build_loss_function_from_dict,
    build_history_object,
    calculate_accuracy,
    print_training_progress,
    print_validation_progress,
)
from neuralpy.layers.linear import Dense
from neuralpy.layers.activation_functions import GELU
from neuralpy.models import Sequential
from neuralpy.optimizer import Adam
from neuralpy.loss_functions import MSELoss

from torch.nn import Linear, GELU as _GELU
from torch.optim import Adam as _Adam
from torch.nn import MSELoss as _MSELoss
from torch import Tensor


def test_generate_layer_name():
    assert generate_layer_name("Dense", 1) == "dense_layer_2"
    assert generate_layer_name("ReLU", 1) == "relu_layer_2"
    assert generate_layer_name("Softmax", 1) == "softmax_layer_2"


def test_build_layer_from_dict():
    layers = []

    layers.append(Dense(n_nodes=32, n_inputs=32))
    layers.append(Dense(n_nodes=32, n_inputs=32))
    layers.append(Dense(n_nodes=32, n_inputs=32))
    layers.append(GELU())

    model_layers = build_layer_from_dict(layers)

    for index, (layer_name, layer) in enumerate(model_layers):
        assert layer_name == f"dense_layer_{index+1}" or f"gelu_layer_{index+1}"

        if "dense" in layer_name:
            assert isinstance(layer, Linear) is True
        else:
            assert isinstance(layer, _GELU) is True


def test_build_optimizer_from_dict():
    model = Sequential()

    model.add(Dense(n_nodes=32, n_inputs=32))
    model.add(Dense(n_nodes=32))

    model.build()

    pytorch_model = model.get_model()

    optimizer, optimizer_arguments = build_optimizer_from_dict(
        Adam(), pytorch_model.parameters()
    )

    assert isinstance(optimizer_arguments, dict) is True

    assert isinstance(optimizer, _Adam) is True


def test_build_loss_function_from_dict():
    loss = MSELoss()

    loss_function, loss_function_arguments = build_loss_function_from_dict(loss)

    assert isinstance(loss_function, _MSELoss) is True

    assert isinstance(loss_function_arguments, dict) is True


def test_build_history_object():
    history = ["loss"]

    history_obj = build_history_object(history)

    assert isinstance(history_obj, dict) is True

    assert isinstance(history_obj["epochwise"], dict) is True
    assert isinstance(history_obj["batchwise"], dict) is True

    assert isinstance(history_obj["epochwise"]["training_loss"], list) is True
    assert isinstance(history_obj["epochwise"]["validation_loss"], list) is True
    assert isinstance(history_obj["batchwise"]["training_loss"], list) is True
    assert isinstance(history_obj["batchwise"]["validation_loss"], list) is True


def test_calculate_accuracy1():
    y = Tensor([1, 1, 2, 2, 1]).view(-1, 1)
    y_pred = Tensor(
        [
            [0.98, 0.12, 0.12],
            [0.12, 0.98, 0.11],
            [0.12, 0.12, 0.99],
            [0.12, 0.12, 0.99],
            [0.12, 0.99, 0.1],
        ]
    )

    score = calculate_accuracy(y, y_pred)

    assert score == 4


def test_calculate_accuracy2():
    y = Tensor([1, 1, 2, 2, 1]).view(-1, 1)
    y_pred = Tensor(
        [
            [0.98, 0.12, 0.12],
            [0.12, 0.11, 0.91],
            [0.12, 0.12, 0.99],
            [0.12, 0.12, 0.99],
            [0.12, 0.99, 0.1],
        ]
    )

    score = calculate_accuracy(y, y_pred)

    assert score == 3


def test_print_training_progress():
    print_training_progress(5, 10, 12, 20, 100, 20, 12)


def test_print_validation_progress():
    print_validation_progress(0.23423, 100, 43)
    print_validation_progress(23, 100)
    print_validation_progress(None, 100)
