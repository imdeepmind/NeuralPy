import pytest
from neuralpy.callbacks import TrainLogger

def test_train_logger_should_throw_type_error():
    with pytest.raises(TypeError) as ex:
        x = TrainLogger()

@pytest.mark.parametrize('path', [(None), (12), (False)])
def test_train_logger_should_throw_value_error(path):
    with pytest.raises(ValueError) as ex:
        x = TrainLogger(path=path)

