"""This algorithm is an implementation of algorithms in the following paper:
    "Clustering Mixed Numeric and Categorical Data: A Cluster Ensemble\ 
    Approach"
"""

import numpy as np

def get_support(data,feature_id,feature_val,cluster):
    """This function compute support for a given value
    """
    n_cluster_size = len(cluster)
    num = 0
    for j in range(n_cluster_size):
        if data[cluster[j],feature_id] == feature_val:
            num = num+1
    return num

def similarity_instance_cluster(data,instance_id,cluster):
    """This function computes the similarity between a new instance
    data[instance_id] and a cluster specified by cluster_id

    Parameters
    ----------
    data: array, shape(n_instances,n_features)
        matrix containing original data

    instance_id: int
        row number of the new instance

    cluster: list
        a list containing the ids of instances in this cluster
    
    Returns
    -------
    sim: float
        the similarity between the input instance and input cluster
    """
    n_instances,n_features = data.shape
    sim = 0.0

    for i in range(n_features):
        
        unique = []
        for j in range(len(cluster)):
            if data[cluster[j],i] not in unique:
                unique.append(data[cluster[j],i])
        temp = 0
        for j in range(len(unique)):
            temp = temp+get_support(data,i,unique[j],cluster)
        sim = sim+get_support(data,i,data[instance_id,i],cluster)*1.0/temp
    return sim

def squeezer(data,thre):
    """This function implements squeezer algorithm base on the paper "Squezzer
    : An Efficient Algorithm for Clustering Categorical Data"
    
    Parameters
    ----------
    data: array, shape(n_instances,n_features)
        the original data that need to be clustered, note that we donnot have
        to specify the number of clusters here

    thre: threshold used to decide if creating a new cluster is necessary

    Returns
    -------
    label: list, length(n_instances)
        label for every instance, label is a list of lists,list[i] represents
        cluster i, list[i] is a list containing the instances ID of cluster i
    """
    # Initialize the clustering result
    label = [[0]]
    
    # Obtain the number of instances and features from input data
    n_instances,n_features = data.shape

    for i in range(1,n_instances):

        # Current number of clusters
        n_clusters = len(label)
        sim = [0]*n_clusters
        # Compute similarity between data[i,:] and each cluster
        for j in range(n_clusters):
            sim[j] = similarity_instance_cluster(data,i,label[j])
        
        sim_max = max(sim)

        for j in range(n_clusters):
            if sim[j] == sim_max:
                sim_max_cluster_id = j

        if sim_max>=thre:
            label[sim_max_cluster_id].append(i)
        else:
            label.append([i])

    return label

if __name__ == "__main__":
    data = np.array([[1,1,1,85,'Biochemistry',5],\
                     [2,1,5,94,'Admission',12],\
                     [3,2,2,92,'DB',2],\
                     [4,3,2,98,'DB',2],\
                     [5,4,3,98,'AI',2],\
                     [6,5,1,98,'Biochemistry',2],\
                     [7,6,5,88,'Admission',12]])

    cluster = [2,3,4]
    instance_id = 5
    thre = 2
    print similarity_instance_cluster(data,instance_id,cluster)
    print squeezer(data,thre)
