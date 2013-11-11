import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def entropy_metric(data, beta=10., mu=0.5):
    tmp = euclidean_distances(data)
    row_sums = tmp.sum(axis=1)
    mtr_distance = tmp*1./row_sums[:,np.newaxis]
    assert mtr_distance.shape[0] == data.shape[0]
    entropy_1 = (np.exp(beta*mtr_distance)-1.)/(np.exp(beta*mu)-1.);
    entropy_2 = (np.exp(beta*(1.-mtr_distance))-1.)/(np.exp(beta*(1.-mu))-1.)
    tp_1 = mtr_distance<mu
    entropy = entropy_1*tp_1+entropy_2*(1-tp_1)

    return np.sum(entropy)

def entropy_metric_original(data):
    tmp = euclidean_distances(data)
    row_sums = tmp.sum(axis=1)
    d = (tmp*1./row_sums[:,np.newaxis]*0.99)+0.01
    assert d.shape[0] == data.shape[0]
    entropy = -np.sum(d*np.log(d)+(1.-d)*np.log(1.-d))
    return entropy

def entropy_metric_1(data):
    tmp = euclidean_distances(data)
    row_sums = tmp.sum(axis=1)
    d = (tmp*1./row_sums[:,np.newaxis])
    print d
    assert d.shape[0] == data.shape[0]
    entropy = 0.
    for i in range(d.shape[0]-1):
        for j in range(i+1,d.shape[0]):
            entropy += -d[i,j]*np.log(d[i,j])+(d[i,j]-1)*np.log(1-d[i,j])
    return entropy

if __name__ == "__main__":
    import numpy as np

    # data with clustering structure
    data_1 = np.array([[1.,1.],
                       [1.,0.9],
                       [-1.,-1.],
                       [-1.,-0.9]])
    # data without clustering structure
    data_2 = np.array([[1.,1.],
                       [1.,0.9],
                       [0.9,1.],
                       [0.9,0.9]])
    entropy_1 = entropy_metric(data_1)
    entropy_2 = entropy_metric(data_2)
    print entropy_1,entropy_2
