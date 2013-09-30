import numpy as np

def similarity_between_clusters(mtr_sim,clusters,method):
    """ Compute pairwise similarity between clusters

    Parameters
    ----------
    mtr_sim: array, shape(n_instances,n_instances)
        similarity matrix
    
    clusters: list, len(n_clusters)
        case-ids for all the clusters 

    method: string, {'average','single','complete'}
        specify linkage criteria, ie, the similarity metric between two
        clusters

    Return 
    ------
    sim_clusters: array, shape(n_clusters,n_clusters)
        matrix containing pairwise similarity between clusters
    """
    n_clusters = len(clusters)
    n_instances = mtr_sim.shape[0]
    sim_clusters = np.zeros((n_clusters,n_clusters))
    for i in range(n_clusters-1):
        for j in range(i+1,n_clusters):
            size_cluster_i = len(clusters[i])
            size_cluster_j = len(clusters[j])
            # pairwise similarity between samples in cluster i and cluster j 
            tmp = np.zeros((size_cluster_i,size_cluster_j))
            for ii in range(size_cluster_i):
                for jj in range(size_cluster_j):
                    tmp[ii,jj] = mtr_sim[clusters[i][ii],clusters[j][jj]]
            if method == 'average':
                sim_clusters[i,j] = np.mean(tmp)
            if method == 'single':
                sim_clusters[i,j] = np.max(tmp)
            if method == 'complete':
                sim_clusters[i,j] = np.min(tmp)
            sim_clusters[j,i] = sim_clusters[i,j]
    return sim_clusters
    
def my_hierarchical_clustering(mtr_sim,method='average'):
    """Hierarchial/Agglomerative clustering given similarity values instead of
    distance values

    Parameters
    ----------
    mtr_sim: array, shape(n_instances,n_instances)
        similarity matrix

    method: string, {'average','single','complete'}
        specify linkage criteria, ie, the similarity metric between two
        clusters

    Returns
    -------
    mtr_lin: array, shape(n_instances-1,4)
        linkage matrix
    """

    # Obtain the number of instances
    n_instances = mtr_sim.shape[0]
    
    # Initialize the linkage matrix
    mtr_lin = np.zeros((n_instances-1,4))
    
    # Initialize the number of clusters
    n_clusters = n_instances

    # Labels for each cluster
    cluster_labels = range(n_clusters)

    # Store sample ids for each cluster
    clusters = []
    for i in range(n_clusters):
        clusters.append([i])
    
    for k in range(n_instances-1):
        # Compute pairwise similarity between clusters
        sim_clusters = np.zeros((n_clusters,n_clusters))
        sim_clusters = similarity_between_clusters(mtr_sim,clusters,method)
        
        # Find the maximal element from the matrix sim_clusters
        max_sim_clusters = np.max(sim_clusters)
        
        # Find the cluster ids that need to be merged
        for i in range(n_clusters-1):
            flag = False
            for j in range(i+1,n_clusters):
                if max_sim_clusters == sim_clusters[i,j]:
                    cluster_merge = [i,j]
                    flag = True
                    break
            if flag == True:
                break
        # Record changes
        mtr_lin[k,0] = int(cluster_labels[cluster_merge[0]])
        mtr_lin[k,1] = int(cluster_labels[cluster_merge[1]])
        mtr_lin[k,2] = max_sim_clusters
        mtr_lin[k,3] = len(clusters[cluster_merge[0]])+\
                len(clusters[cluster_merge[1]])

        # Merge clusters
        clusters[cluster_merge[0]] += clusters[cluster_merge[1]]
        del clusters[cluster_merge[1]]
        
        # Modify cluster labels accordingly
        cluster_labels[cluster_merge[0]] = n_instances+k
        del cluster_labels[cluster_merge[1]]

        # Decrease the number of clusters
        n_clusters -= 1
    
    return mtr_lin

if __name__ == "__main__":
    from sklearn.metrics.pairwise import rbf_kernel
    from scipy.cluster.hierarchy import linkage
    from scipy.cluster.hierarchy import dendrogram
    import matplotlib.pyplot as plt

    data = np.array([[1.0,1.0],
                     [1.2,1.0],
                     [1.0,0.9],
                     [0.0,-0.5],
                     [0.0,-1.0],
                     [-1.0,-1.0]])
    mtr_sim = rbf_kernel(data,data)
    mtr_lin_1 = my_hierarchical_clustering(mtr_sim,method='complete')

    for i in range(mtr_lin_1.shape[0]):
        if mtr_lin_1[i,0]>mtr_lin_1[i,1]:
            tp = mtr_lin_1[i,0]
            mtr_lin_1[i,0] = mtr_lin_1[i,1]
            mtr_lin_1[i,1] = tp

    tmp = []
    for i in range(mtr_sim.shape[0]-1):
        for j in range(i+1,mtr_sim.shape[1]):
            tmp.append(-mtr_sim[i,j])
    mtr_lin_2 = linkage(tmp,method='complete')
    fig_1 = plt.figure(1,figsize=(30,30)) 
    dend = dendrogram(mtr_lin_1,orientation='top')
    plt.savefig('Dendrogram.png')

    # Output the result
    for i in range(mtr_lin_1.shape[0]):
        print int(mtr_lin_1[i,0]),int(mtr_lin_1[i,1]),mtr_lin_1[i,2],\
                int(mtr_lin_1[i,3])
        print int(mtr_lin_2[i,0]),int(mtr_lin_2[i,1]),mtr_lin_2[i,2],\
                int(mtr_lin_2[i,3])
