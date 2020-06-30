import pytest
from torch.nn import Dropout3d as _Dropout3D
from neuralpy.regulariziers import Dropout3D

@pytest.mark.parametrize(
	"p, name", 
	[
		(6.3, False),
		(-4.2, False),
		(.33, False),
		(.56, 12)
	]
)
def test_dense_should_throw_value_error(p, name):
    with pytest.raises(ValueError) as ex:
        x = Dropout3D(p=p, name=name)

@pytest.mark.parametrize(
	"p, name", 
	[
		(.3, "test"),
		(.2, None)
	]
)
def test_dense_get_layer_method(p, name):
	x = Dropout3D(p=p, name=name)

	assert x.get_input_dim(12, "dense") == None
		
	details = x.get_layer()

	assert isinstance(details, dict) == True

	assert details["layer_details"] == None

	assert details["name"] == name

	assert issubclass(details["layer"], _Dropout3D) == True

	assert isinstance(details["keyword_arguments"], dict) == True

	assert details["keyword_arguments"]["p"] == p

	assert details["keyword_arguments"]["inplace"] == False

def test_dense_get_layer_method_wit_no_parameter():
	x = Dropout3D()

	assert x.get_input_dim(12, "dense") == None
		
	details = x.get_layer()

	assert isinstance(details, dict) == True

	assert details["layer_details"] == None

	assert details["name"] == None

	assert issubclass(details["layer"], _Dropout3D) == True

	assert isinstance(details["keyword_arguments"], dict) == True

	assert details["keyword_arguments"]["p"] == .5

	assert details["keyword_arguments"]["inplace"] == False