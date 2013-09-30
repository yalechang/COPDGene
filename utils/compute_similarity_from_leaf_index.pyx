import numpy as np
cimport numpy as np

ctypedef np.float64_t dtype_t

def compute_similarity_from_leaf_index(np.ndarray[dtype_t,ndim=2] leaf_index):
    """This function computes similarity matrix from leaf index that samples
    end up in.

    Parameters
    ----------
    leaf_index: array, shape(n_instances,n_estimators)
        For each datapoint x in dataset and for each tree(estimator) in the
        forest, store index of the leaf x ends up in

    Returns
    -------
    mtr_affinity: array, shape(n_instances,n_instances)
        mtr_affinity[i,j] represents the similarity between sample i and sample
        j, which is measured by the frequency that sample i and sampe j end up
        in the same leaf in a tree.
    """
    # Obtain the number of samples and estimators
    cdef int n_instances = leaf_index.shape[0]
    cdef int n_estimators = leaf_index.shape[1]

    # Initialize the similarity matrix to be returned
    cdef np.ndarray[dtype_t,ndim=2] mtr_affinity = np.zeros((n_instances,n_instances))
    
    cdef int i,j,m,n

    for j in range(n_estimators):
        for m in range(n_instances-1):
            for n in range(m+1,n_instances):
                if leaf_index[m,j] == leaf_index[n,j]:
                    mtr_affinity[m,n] += 1
    
    # Make the affinity matrix symmetric
    for i in range(n_instances-1):
        for j in range(i+1,n_instances):
            mtr_affinity[i,j] = mtr_affinity[i,j]*1./n_estimators
            mtr_affinity[j,i] = mtr_affinity[i,j]
    return mtr_affinity

