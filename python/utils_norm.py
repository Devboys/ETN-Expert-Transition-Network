import numpy as np
from typing import Iterable

__all__ = ['norm_params', 'normalize_vectors', 'denormalize_vectors', 'denormalize_single', 'normalize_single']


def norm_params(vectors: Iterable[np.ndarray]):
    if len(vectors.shape) == 2:
        mean, std = vectors.mean(axis=0, dtype=np.float32), vectors.std(axis=0, dtype=np.float32)
    else:
        mean, std = np.concatenate(vectors).mean(axis=0, dtype=np.float32), \
                    np.concatenate(vectors).std(axis=0, dtype=np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def normalize_vectors(vectors: Iterable[np.ndarray], mean, std):
    """
    Normalizes every vector in the given vectors-array.

    :param vectors: The vectors to normalize
    :param mean: The mean to normalize with
    :param std: the standard deviation to normalize with
    :return: The normalized vectors
    """

    return [normalize_single(vector, mean, std) for vector in vectors]


def denormalize_vectors(vectors: Iterable[np.ndarray], mean, std):
    """
    Denormalizes every vector in the given vectors-array.

    :param vectors: The vectors to denormalize
    :param mean: The mean that was used to normalize the data
    :param std: The standard deviation that was used to normalize the dat
    :return: The denormalized vectors
    """

    return [denormalize_single(vector, mean, std) for vector in vectors]


def normalize_single(val, mean, std):
    """
    Normalizes a single value through Standardization.

    :param val: The value to normalize
    :param mean: The mean of the dataset
    :param std: The standard deviation of the dataset
    :return: The normalized value
    """

    return (val - mean) / std


def denormalize_single(val, mean, std):
    """
    Denormalizes a single value through Standardization.

    :param val: The value to denormalize
    :param mean: The mean that was used to normalize the data
    :param std: The standard deviation that was used to normalize the dat
    :return: The denormalized value
    """

    return (val * std) + mean