import pytest
from neuralpy.callbacks import TrainLogger


def test_train_logger_should_throw_type_error():
    with pytest.raises(TypeError):
        TrainLogger()


@pytest.mark.parametrize("path", [None, 12, False])
def test_train_logger_should_throw_value_error(path):
    with pytest.raises(ValueError):
        TrainLogger(path=path)


def test_train_logger_should_not_throw_any_error():
    x = TrainLogger(path="ignore")
    x.callback(
        10,
        1,
        {},
        {},
        {},
        None,
    )
    x.callback(
        10,
        1,
        {"inplace": True},
        {"inplace": True},
        {"inplace": True},
        None,
    )
