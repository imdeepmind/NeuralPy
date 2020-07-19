import pytest
from torch.nn import RNNCell as _RNNCell
from neuralpy.layers import RNNCell


def test_RNN_should_throws_type_error():
    with pytest.raises(TypeError) as ex:
        x = RNNCell()