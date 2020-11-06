"""Specific instantiations of memory models that use the full history."""
from . import retrieval


def episodic_64dk_ret4_half_stride(get_transformer):
    prediction_network = get_transformer(name="prediction")
    query_network = get_transformer(name="query")
    key_network = get_transformer(name="key")

    sequence_length = 32
    key_size = 64
    history_stride = sequence_length // 2
    num_retrieved = 4

    return retrieval.EpisodicRetriever(
        prediction_network=prediction_network,
        key_network=key_network,
        query_network=query_network,
        key_size=key_size,
        history_stride=history_stride,
        num_retrieved=num_retrieved,
    )


def episodic_32dk_ret4_half_stride(get_transformer):
    prediction_network = get_transformer(name="prediction")
    query_network = get_transformer(name="query")
    key_network = get_transformer(name="key")

    sequence_length = 32
    key_size = 32
    history_stride = sequence_length // 2
    num_retrieved = 4

    return retrieval.EpisodicRetriever(
        prediction_network=prediction_network,
        key_network=key_network,
        query_network=query_network,
        key_size=key_size,
        history_stride=history_stride,
        num_retrieved=num_retrieved,
    )


def no_history_64dk_ret4_half_stride(get_transformer):
    prediction_network = get_transformer()
    return retrieval.NoHistoryWrapper(memory_component=prediction_network)


def no_history_32dk_ret4_half_stride(get_transformer):
    prediction_network = get_transformer()
    return retrieval.NoHistoryWrapper(memory_component=prediction_network)
