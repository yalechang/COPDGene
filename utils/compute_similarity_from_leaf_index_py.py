import numpy as np

def compute_similarity_from_leaf_index_py(leaf_index):
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
    n_instances,n_estimators = leaf_index.shape

    # Initialize the similarity matrix to be returned
    mtr_affinity = np.zeros((n_instances,n_instances))

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

def test_compute_similarity_from_leaf_index():
    leaf_index = np.array([[1,1,1,3],
                           [2,3,1,1],
                           [1,1,3,2],
                           [1,2,2,3],
                           [2,3,2,1],
                           [3,2,1,2]])
    mtr_affinity = compute_similarity_from_leaf_index(leaf_index)
    return mtr_affinity

if __name__ == "__main__":
    print test_compute_similarity_from_leaf_index()

