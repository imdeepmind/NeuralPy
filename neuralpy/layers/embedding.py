"""Embedding layer for NeuralPy"""

from torch.nn import Embedding as _Embedding


class Embedding:
    """
        A simple lookup table that stores embeddings of a fixed dictionary and size
        To learn more about RNN, please check pytorch
        documentation at https://pytorch.org/docs/stable/nn.html#embedding

        Supported Arguments:
            num_embeddings: (Integer) size of the dictionary of embeddings
            embedding_dim: (Integer) the size of each embedding vector
            padding_idx: (Integer) If given, pads the output with the
                embedding vector at padding_idx (initialized to zeros)
                whenever it encounters the index
            max_norm: (Float) If given, each embedding vector with
                norm larger than max_norm is renormalized to have norm max_norm
            norm_type: (Float) The p of the p-norm to compute for the max_norm option.Default 2
            scale_grad_by_freq: (Boolean) If given, this will scale gradients by the
                inverse of frequency of the words in the mini-batch. Default False
            sparse: (Boolean) If True, gradient w.r.t. weight matrix will be a sparse tensor.
    """

    def __init__(
            self, num_embeddings, embedding_dim, padding_idx=None,
            max_norm=None, norm_type=2.0, scale_grad_by_freq=False,
            sparse=False, name=None):
        """
            __init__ method for Embedding
            Supported Arguments:
                num_embeddings: (Integer) size of the dictionary of embeddings
                embedding_dim: (Integer) the size of each embedding vector
                padding_idx: (Integer) If given, pads the output with the
                    embedding vector at padding_idx (initialized to zeros)
                    whenever it encounters the index
                max_norm: (Float) If given, each embedding vector with
                    norm larger than max_norm is renormalized to have norm max_norm
                norm_type: (Float) The p of the p-norm to compute for the max_norm option.Default 2
                scale_grad_by_freq: (Boolean) If given, this will scale gradients by the
                    inverse of frequency of the words in the mini-batch. Default False
                sparse: (Boolean) If True, gradient w.r.t. weight matrix will be a sparse tensor.

        """

        # Checking num_embeddings
        if not num_embeddings or not isinstance(num_embeddings, int):
            raise ValueError("Please provide a valid  num_embeddings")

        # Checking embedding_dim
        if not embedding_dim or not isinstance(embedding_dim, int):
            raise ValueError("Please provide a valid  embedding_dim")

        # Checking padding_idx, It is an optional field
        if padding_idx is not None and not isinstance(padding_idx, int):
            raise ValueError("Please provide a valid padding_idx")

        # Checking max_norm, It is an optional field
        if max_norm is not None and not isinstance(max_norm, float):
            raise ValueError("Please provide a valid max_norm")

        # Checking norm_type, It is an optional field
        if not norm_type or not isinstance(norm_type, float):
            raise ValueError("please provide a valid norm_type")

        # Checking scale_grad_by_freq, It is an optional field
        if not isinstance(scale_grad_by_freq, bool):
            raise ValueError("Please provide a valid scale_grad_by_freq")

        # Checking sparse, It is an optional field
        if not isinstance(sparse, bool):
            raise ValueError("Please provide a valid sparse")

        # Checking name, It is an optional field
        if name is not None and not (isinstance(name, str) and name):
            raise ValueError("Please provide a valid name")

        # Storing values

        self.__num_embeddings = num_embeddings
        self.__embedding_dim = embedding_dim

        self.__padding_idx = padding_idx
        self.__max_norm = max_norm
        self.__norm_type = norm_type
        self.__scale_grad_by_freq = scale_grad_by_freq
        self.__sparse = sparse
        self.__name = name

    # pylint: disable=W0613,R0201
    def get_input_dim(self, prev_input_dim, prev_layer_type):
        """
            This method calculates the input shape for layer based on previous output layer.

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # Embedding does not need to n_input, so returning None
        return None

    def get_layer(self):
        """
            This method returns the details as dict of the layer.

            This method is used by the NeuralPy Models, for building the models.
            No need to call this method for using NeuralPy.
        """
        # Returning all the details of the layer
        return{
            'layer_details': (self.__embedding_dim, ),
            'name': self.__name,
            'type': 'Embedding',
            'layer': _Embedding,
            'keyword_arguments': {
                'num_embeddings': self.__num_embeddings,
                'embedding_dim': self.__embedding_dim,
                'padding_idx': self.__padding_idx,
                'max_norm': self.__max_norm,
                'norm_type': self.__norm_type,
                'scale_grad_by_freq': self.__scale_grad_by_freq,
                'sparse': self.__sparse
            }
        }
