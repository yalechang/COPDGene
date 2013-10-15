#from math import log
#from scipy.misc import comb
#from scipy.special import gammaln

import numpy as np
#from sklearn.utils.fixes import unique
from sklearn.metrics.cluster.supervised import *

def nmi_revised(labels_true, labels_pred):
    """Normalized Mutual Information between two clusterings

    Normalized Mutual Information (NMI) is an normalization of the Mutual
    Information (MI) score to scale the results between 0 (no mutual
    information) and 1 (perfect correlation).

    This measure is not adjusted for chance. Therefore
    ``adjusted_mustual_info_score`` might be preferred.

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is furthermore symmetric: switching `label_true` with
    `label_pred` will return the same score value. This can be useful to
    measure the agreement of two independent label assignments strategies
    on the same dataset when the real ground truth is not known.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        A clustering of the data into disjoint subsets.

    labels_pred : array, shape = [n_samples]
        A clustering of the data into disjoint subsets.

    Returns
    -------
    nmi: float
       score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

    See also
    --------
    adjusted_rand_score: Adjusted Rand Index
    adjusted_mutual_info_score: Adjusted Mutual Information (adjusted
        against chance)

    Examples
    --------

    Perfect labelings are both homogeneous and complete, hence have
    score 1.0::

      >>> from sklearn.metrics.cluster import normalized_mutual_info_score
      >>> normalized_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1])
      1.0
      >>> normalized_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    If classes members are completly splitted across different clusters,
    the assignment is totally in-complete, hence the NMI is null::

      >>> normalized_mutual_info_score([0, 0, 0, 0], [0, 1, 2, 3])
      0.0

    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    # Special limit cases: no clustering since the data is not split.
    # This is a perfect match hence return 1.0.
    if (classes.shape[0] == clusters.shape[0] == 1
        or classes.shape[0] == clusters.shape[0] == 0):
        return 1.0
    contingency = contingency_matrix(labels_true, labels_pred)
    contingency = np.array(contingency, dtype='float')
    # Calculate the MI for the two clusterings
    mi = mutual_info_score(labels_true, labels_pred,
                           contingency=contingency)
    # Calculate the expected value for the mutual information
    # Calculate entropy for each labeling
    h_true, h_pred = entropy(labels_true), entropy(labels_pred)
    nmi = mi / max(h_true, h_pred)
    return nmi
