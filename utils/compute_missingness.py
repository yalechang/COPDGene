def compute_missingness(data):
    """This function compute the number of missing values for every feature
    in the given dataset

    Parameters
    ----------
    data: array, shape(n_instances,n_features)
        array containing the dataset, which might contain missing values

    Returns
    -------
    n_missing: list, len(n_features)
        list containing the number of missing values for every feature
    """
    n_instances,n_features = data.shape
    
    n_missing = [0]*n_features

    for j in range(n_features):
        for i in range(n_instances):
            if data[i,j] == '':
                n_missing[j] += 1
    return n_missing

def test_compute_missingness():
    import numpy as np
    data = np.empty((4,9),dtype=list)
    data[0,0] = ''
    data[0,1] = ''
    data[1,4] = ''
    for i in range(6):
        data[3,i] = ''
    n_missing = compute_missingness(data)
    print n_missing

if __name__ == "__main__":
    test_compute_missingness()

