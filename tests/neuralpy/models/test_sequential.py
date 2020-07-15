from neuralpy.models import Sequential
from neuralpy.layers import Dense

import pytest

def test_add_method():
    model = Sequential()

    model.add(Dense(n_nodes=32, n_inputs=45))

    model.build()

    with pytest.raises(Exception) as ex:
        model.add(Dense(n_nodes=32, n_inputs=45))
