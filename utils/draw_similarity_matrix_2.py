import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from pylab import *

def draw_similarity_matrix(data,labels,n_clusters,sigma_rbf=1.0):
    """
    affinity: similarity matrix
        shape(n_samples,n_samples)
    labels: list
        length(n_samples)
    n_clusters: int
        number of clusters
    """
    n_samples,n_features = data.shape
    counter = [0]*n_clusters
    index_start = range(n_clusters)

    for i in range(n_samples):
        for j in range(n_clusters):
            if labels[i] == j:
                counter[j] = counter[j]+1
    for i in range(1,n_clusters):
        index_start[i] = sum(counter[0:i])

    #num is the number of elements in a cluster
    num = [0]*n_clusters

    #data_order is the new data after ordering input data
    data_order = np.zeros((n_samples,n_features))
    for i in range(n_samples):
        cluster_id = labels[i]
        i_new = index_start[cluster_id]+num[cluster_id]
        for j in range(n_features):
            data_order[i_new,j] = data[i,j]
        num[cluster_id] = num[cluster_id]+1
    
    affinity = pairwise_kernels(data_order,data_order,metric='rbf'
                               ,gamma=1.0/(2*sigma_rbf**2))
    for i in range(n_samples):
        affinity[i,i] = 0
    imshow(affinity,interpolation='nearest')
    grid(True)
    #return counter,index_start
    #return data_order


if __name__ == "__main__":
    data = np.arange(36).reshape(9,4)
    labels = [0,1,1,0,2,2,1,0,2]
    n_clusters = 3
    draw_similarity_matrix(data,labels,n_clusters)

"""
a = np.random.normal(0.0,0.5,size=(5000,10))**2
a = a/np.sum(a,axis=1)[:,None]
pcolor(a)
maxvi = np.argsort(a,axis=1)
ii = np.argsort(maxvi[:,-1])
pcolor(a[ii,:])
show()
"""

"""
A = rand(20,20)
figure(1)
imshow(A,interpolation='nearest')
grid(True)
show()
"""


