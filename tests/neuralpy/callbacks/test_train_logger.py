import pytest
from neuralpy.callbacks import TrainLogger

# The test scripts may need to be updated


def test_train_logger_should_throw_type_error():
    with pytest.raises(TypeError) as ex:
        x = TrainLogger()


@pytest.mark.parametrize('path', [(None), (12), (False)])
def test_train_logger_should_throw_value_error(path):
    with pytest.raises(ValueError) as ex:
        x = TrainLogger(path=path)


def test_train_logger_should_not_throw_any_error():
    x = TrainLogger(path='ignore')
    x.callback(10, 1, {}, {}, {}, None)
    x.callback(10,
               1,
               {
                   'inplace': True
               },
               {
                   'inplace': True
               },
               {
                   'inplace': True
               },
               None)
