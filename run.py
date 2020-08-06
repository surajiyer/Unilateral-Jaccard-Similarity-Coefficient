from typing import Iterable, Tuple
from python_data_utils.sklearn.cluster import ap_precomputed
import random
import string


def ap_ujaccard(items: Iterable[Tuple[int, set]],
                depth: int = 3, n_jobs: int = 1,
                verbose: bool = True, **kwargs) -> dict:
    """
    Cluster text documents with affinity propagation
    based on Unilateral Jaccard similarity scores.

    :param items: Iterable of tuples of type [(int, set),...]
        Each tuple in list of items is a pair of item id (int) and item (set).
    :param depth: int
    :param n_jobs: int
        Number of processes to parallelize computation of the similarity matrix.
    :param verbose: bool
    :param kwargs: Additional arguments for sklearn.cluster.AffinityPropagation
    :return: dict
    """
    assert isinstance(depth, int) and depth > 0,\
        'depth must be a positive integer.'
    assert isinstance(verbose, bool), 'verbose must be a boolean.'

    # Computer unilateral Jaccard similarity between documents
    from .unilateral_jaccard import ujaccard_similarity_score, calculate_edges_list
    V = [doc for _, doc in items]
    E = calculate_edges_list(V)
    uJaccard_similarity = ujaccard_similarity_score(
        (range(len(V)), E), depth=depth, n_jobs=n_jobs)
    return ap_precomputed(
        uJaccard_similarity, verbose, **kwargs)


if __name__ == '__main__':
    N = 10
    n_samples = 50
    test_data = []
    for i in range(n_samples):
        test_data.append((i, set(
            ''.join(random.choice(string.ascii_uppercase[:5]) for _ in range(N)))))
    clusters = ap_ujaccard(test_data)
    assert isinstance(clusters, dict)
