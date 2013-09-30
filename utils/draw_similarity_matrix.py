import numpy as np
from pylab import *
import copy

def draw_similarity_matrix(affinity,labels,n_clusters):
    """This function permutes the affinity matrix according to clustering
    results specified by labels so that the matrix will show block structure.
    It could be used to evaluate the performance of clustering result.

    Parameters
    ----------
    affinity: array, shape(n_samples,n_samples)
        similarity matrix 
        
    labels: list, len(n_samples)
        clustering result

    n_clusters: int
        number of clusters

    Returns
    -------
    pos_old: dict, len(n_samples)
        original row index of rows in the new matrix
    """
    # Get the number of samples 
    n_samples = len(labels)

    # number of samples in every cluster
    counter = [0]*n_clusters
    for i in range(n_samples):
        for j in range(n_clusters):
            if labels[i] == j:
                counter[j] = counter[j]+1
    
    # rank the clusters according to their cluster size
    new_counter = copy.copy(counter)
    new_cluster_id = range(n_clusters)
    for i in range(0,n_clusters-1):
        for j in range(i+1,n_clusters):
            if new_counter[i]<new_counter[j]:
                tmp = new_counter[i]
                new_counter[i] = new_counter[j]
                new_counter[j] = tmp
                tmp1 = new_cluster_id[i]
                new_cluster_id[i] = new_cluster_id[j]
                new_cluster_id[j] = tmp1
    # the index of the 1st sample of every cluster
    index_start = range(n_clusters)
    for i in range(1,n_clusters):
        index_start[i] = sum(new_counter[0:i])

    # num is the number of elements in a cluster
    # it changes every time, no greater than counter[j] 
    num = [0]*n_clusters
    
    # pos_old contain the original location of the new location
    pos_old = {}
       
    # affinity_order is the new affinity matrix after ordering rows
    affinity_order = np.zeros((n_samples,n_samples))
    affinity_1 = np.zeros((n_samples,n_samples))
    for i in range(n_samples):
        for j in range(len(new_cluster_id)):
            if new_cluster_id[j] == labels[i]:
                tmp = j
                break
        i_new = index_start[tmp]+num[tmp]
        pos_old[i_new] = i
        affinity_order[i_new,:] = affinity[i,:]
        num[tmp] = num[tmp]+1
    
    # affinity_1 is the new affinity matrix after ordering columns 
    num = [0]*n_clusters
    for j in range(n_samples):
        for i in range(len(new_cluster_id)):
            if new_cluster_id[i] == labels[j]:
                tmp = i
                break
        j_new = index_start[tmp]+num[tmp]
        affinity_1[:,j_new] = affinity_order[:,j]
        num[tmp] = num[tmp]+1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = imshow(affinity_1,interpolation='nearest')
    #grid(True)
    fig.colorbar(im)
    return ax,pos_old

if __name__ == "__main__":
    from sklearn.metrics.pairwise import rbf_kernel
    import random
    data = np.array([[0,0.1],
                     [1,-1],
                     [1,1],
                     [0.1,0],
                     [1,0.9],
                     [0.9,1],])
    sigma = 1.
    affinity = rbf_kernel(data,gamma=1./(2*sigma**2))
    #print affinity
    x0,x1,x2 = range(3)
    random.shuffle([x0,x1,x2])
    labels = [x0,x2,x1,x0,x1,x1]
    n_clusters = 3
    pos_old = draw_similarity_matrix(affinity,labels,n_clusters)
    show()
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


