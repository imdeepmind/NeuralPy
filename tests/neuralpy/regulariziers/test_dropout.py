import pytest
from torch.nn import Dropout as _Dropout
from neuralpy.regulariziers import Dropout

# Possible values
ps = [6.3, -0.36, 'asd', '', False]
names = [False, 12]

@pytest.mark.parametrize(
	"p, name", 
	[(p, name) for p in ps
		 	   for name in names]
)
def test_dense_should_throw_value_error(p, name):
    with pytest.raises(ValueError) as ex:
        x = Dropout(p=p, name=name)


# Possible values
ps = [0.3, 0.8]
names = ["Test", None]

@pytest.mark.parametrize(
	"p, name", 
	[(p, name) for p in ps
		 	   for name in names]
)
def test_dense_get_layer_method(p, name):
	x = Dropout(p=p, name=name)

	assert x.get_input_dim(12) == None
		
	details = x.get_layer()

	assert isinstance(details, dict) == True

	assert details["n_inputs"] == None

	assert details["n_nodes"] == None

	assert details["name"] == name

	assert issubclass(details["layer"], _Dropout) == True

	assert isinstance(details["keyword_arguments"], dict) == True

	assert details["keyword_arguments"]["p"] == p

	assert details["keyword_arguments"]["inplace"] == False