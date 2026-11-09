import numpy as np
from natural_cifar import SEED, split_indices


def test_natural_split_is_disjoint_stratified_complete_and_reproducible():
    labels = np.tile(np.arange(100), 100)
    practice, evaluation = split_indices(labels)
    assert len(practice) == 2000 and len(evaluation) == 8000
    assert not set(practice) & set(evaluation)
    assert len(set(practice) | set(evaluation)) == 10000
    np.testing.assert_array_equal(np.bincount(labels[practice]), np.full(100, 20))
    np.testing.assert_array_equal(np.bincount(labels[evaluation]), np.full(100, 80))
    p2, e2 = split_indices(labels, SEED)
    np.testing.assert_array_equal(practice, p2)
    np.testing.assert_array_equal(evaluation, e2)
