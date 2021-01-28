import pytest
from torch.nn import Embedding as _Embedding
from neuralpy.layers.sparse import Embedding


def test_embedding_should_throw_type_error():
    with pytest.raises(TypeError):
        Embedding()


@pytest.mark.parametrize(
    "num_embeddings, embedding_dim, padding_idx, \
    max_norm, norm_type, scale_grad_by_freq, sparse, name",
    [
        # Checking num_embeddings validations
        ("invalid", 80, 4, 5.6, 4.5, False, False, None),
        (False, 80, 4, 5.6, 4.5, False, False, None),
        # Checking embedding_dim validations
        (300, "invalid", 4, 5.6, 4.5, False, False, None),
        (300, False, 4, 5.6, 4.5, False, False, None),
        # Checking padding_idx validations
        (300, 80, "invalid", 5.6, 4.5, False, False, None),
        (300, 80, 234.53, 5.6, 4.5, False, False, None),
        # Checking max_norm validations
        (300, 80, 4, False, 4.5, False, False, None),
        (300, 80, 4, 5, 4.5, False, False, None),
        (300, 80, 4, "invalid", 4.5, False, False, None),
        # Checking norm_type validations
        (300, 80, 4, 5.4, False, False, False, None),
        (300, 80, 4, 5.4, 4, False, False, None),
        (300, 80, 4, 5.4, "test", False, False, None),
        # Checking scale_grad_by_freq validations
        (300, 80, 4, 5.4, 5.6, 234, False, None),
        (300, 80, 4, 5.4, 4.4, "test", False, None),
        # Checking sparse validations
        (300, 80, 4, 5.4, 4.4, False, 324, None),
        (300, 80, 4, 5.4, 4.4, False, "test", None),
        # Checking name
        (300, 80, 4, 5.4, 4.4, False, False, False),
        (300, 80, 4, 5.4, 4.4, False, False, ""),
        (300, 80, 4, 5.4, 4.4, False, False, 24),
    ],
)
def test_embedding_should_throw_value_error(
    num_embeddings,
    embedding_dim,
    padding_idx,
    max_norm,
    norm_type,
    scale_grad_by_freq,
    sparse,
    name,
):

    with pytest.raises(ValueError):

        Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            name=name,
        )


# Possible Values
num_embeddingss = [1, 3]
embedding_dims = [2, 1]
padding_idxs = [3, None]
max_norms = [1.0, None]
norm_types = [2.0, 1.0]
scale_grad_by_freqs = [True, False]
sparses = [False, True]
names = ["test", None]


@pytest.mark.parametrize(
    "num_embeddings, embedding_dim, padding_idx, \
    max_norm, norm_type, scale_grad_by_freq, sparse, name",
    [
        (
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            name,
        )
        for num_embeddings in num_embeddingss
        for embedding_dim in embedding_dims
        for padding_idx in padding_idxs
        for max_norm in max_norms
        for norm_type in norm_types
        for scale_grad_by_freq in scale_grad_by_freqs
        for sparse in sparses
        for name in names
    ],
)
def test_embedding_get_layer_method(
    num_embeddings,
    embedding_dim,
    padding_idx,
    max_norm,
    norm_type,
    scale_grad_by_freq,
    sparse,
    name,
):

    x = Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx,
        max_norm=max_norm,
        norm_type=norm_type,
        scale_grad_by_freq=scale_grad_by_freq,
        sparse=sparse,
        name=name,
    )

    prev_input_dim = (6,)

    x.set_input_dim(prev_input_dim, "embedding")

    details = x.get_layer()

    assert isinstance(details, dict) is True

    assert details["layer_details"] == (embedding_dim,)

    assert details["name"] == name

    assert issubclass(details["layer"], _Embedding) is True

    assert isinstance(details["keyword_arguments"], dict) is True

    assert details["keyword_arguments"]["num_embeddings"] == num_embeddings

    assert details["keyword_arguments"]["embedding_dim"] == embedding_dim

    assert details["keyword_arguments"]["padding_idx"] == padding_idx

    assert details["keyword_arguments"]["max_norm"] == max_norm

    assert details["keyword_arguments"]["norm_type"] == norm_type

    assert details["keyword_arguments"]["scale_grad_by_freq"] == scale_grad_by_freq

    assert details["keyword_arguments"]["sparse"] == sparse
