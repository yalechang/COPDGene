import pickle
import numpy as np
from sklearn.cluster.spectral import spectral_clustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster.supervised import normalized_mutual_info_score
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import euclidean_distances
from draw_similarity_matrix import draw_similarity_matrix
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA,KernelPCA

from array_maxmin import array_maxmin
# from nmi_revised import nmi_revised

# number of sets of features
M = 4

# number of clusters
K = 4

# the method used:kmeans, gaussian, linear, euclidean
# Nothe that 'euclidean' would make the program fail
method_id = 'gaussian'

# scale parameter for rbf in spectral clustering for samples
sigma_rbf = 2.6

pkl_file = open("data_gold.pkl","rb")
data_gold = pickle.load(pkl_file)
data = data_gold[0]
gold = data_gold[1]
for i in range(len(gold)):
    gold[i] = gold[i]-1
pkl_file.close()
n_samples,n_features = data.shape

pkl_file = open("matrix_hsic.pkl","rb")
matrix_hsic = pickle.load(pkl_file)
pkl_file.close()

labels_predict = spectral_clustering(matrix_hsic,n_clusters=M)
print labels_predict

plt.figure(0)
draw_similarity_matrix(matrix_hsic,labels_predict,M)

"""
if labels_predict[0] == 0:
    print ['figure 1: whole','figure 2: emphysema','figure 3: airway']
else:
    print ['figure 1: whole','figure 2: airway','figure 3:emphysema']

#the length of two feature sets
len_fs1 = sum(labels_predict)
len_fs2 = len(labels_predict)-len_fs1
#print len_fs1,len_fs2

#the data of two feature sets
data_fs1 = np.zeros((n_samples,len_fs1))
data_fs2 = np.zeros((n_samples,len_fs2))
index_fs1 = 0
index_fs2 = 0

for j in range(n_features):
    if labels_predict[j] == 1:
        for i in range(n_samples):
            data_fs1[i,index_fs1] = data[i,j]
        index_fs1 = index_fs1+1
    else:
        for i in range(n_samples):
            data_fs2[i,index_fs2] = data[i,j]
        index_fs2 = index_fs2+1

if method_id == 'kmeans':
    #==========================CASE 1=================================
    #                          KMeans
    #=================================================================
    # try k-means
    estimator_0 = KMeans(init='random',n_clusters=K,n_init=10,
                         n_jobs=-1)
    estimator_0.fit(data)
    labels_predict = estimator_0.labels_

    estimator_1 = KMeans(init='random',n_clusters=K,n_init=10,
                         n_jobs=-1)
    estimator_1.fit(data_fs1)
    labels_fs1_predict = estimator_1.labels_

    estimator_2 = KMeans(init='random',n_clusters=K,n_init=10,
                         n_jobs=-1)
    estimator_2.fit(data_fs2)
    labels_fs2_predict = estimator_2.labels_
    affinity_fs0 = pairwise_kernels(data,data,metric='rbf'
                                   ,gamma=1.0/(2*sigma_rbf**2))
    affinity_fs1 = pairwise_kernels(data_fs1,data_fs1,metric='rbf'
                                   ,gamma=1.0/(2*sigma_rbf**2))
    affinity_fs2 = pairwise_kernels(data_fs2,data_fs2,metric='rbf'
                                   ,gamma=1.0/(2*sigma_rbf**2))
else:
    if method_id == 'gaussian':
        #============================CASE 2=================================
        #           Spectral Clustering: Gaussian Kernel
        #===================================================================
        # Gaussian kernel
        affinity_fs0 = pairwise_kernels(data,data,metric='rbf'
                                        ,gamma=1.0/(2*sigma_rbf**2))
        affinity_fs1 = pairwise_kernels(data_fs1,data_fs1,metric='rbf'
                                        ,gamma=1.0/(2*sigma_rbf**2))
        affinity_fs2 = pairwise_kernels(data_fs2,data_fs2,metric='rbf'
                                        ,gamma=1.0/(2*sigma_rbf**2))
        for i in range(affinity_fs0.shape[0]):
            affinity_fs0[i,i] = 0
            affinity_fs1[i,i] = 0
            affinity_fs2[i,i] = 0
    elif method_id == 'linear':
        #============================CASE 3====================================
        #           Spectral Clustering: Linear Kernel
        #======================================================================
        # Linear kernel
        affinity_fs0 = pairwise_kernels(data,data,metric='linear')
        affinity_fs1 = pairwise_kernels(data_fs1,data_fs1,metric='linear')
        affinity_fs2 = pairwise_kernels(data_fs2,data_fs2,metric='linear')
        
        affinity_fs0_max = array_maxmin(affinity_fs0,flag='max')
        affinity_fs1_max = array_maxmin(affinity_fs1,flag='max')
        affinity_fs2_max = array_maxmin(affinity_fs2,flag='max')

        for i in range(n_samples):
            for j in range(n_samples):
                affinity_fs0[i,j] = 1.0/(1+np.exp(-affinity_fs0[i,j]))
                #affinity_fs0[i,j] = affinity_fs0[i,j]/affinity_fs0_max
                affinity_fs1[i,j] = 1.0/(1+np.exp(-affinity_fs1[i,j]))
                #affinity_fs1[i,j] = affinity_fs1[i,j]/affinity_fs1_max
                affinity_fs2[i,j] = 1.0/(1+np.exp(-affinity_fs2[i,j]))
                #affinity_fs2[i,j] = affinity_fs2[i,j]/affinity_fs2_max
        for i in range(affinity_fs0.shape[0]):
            affinity_fs0[i,i] = 0
            affinity_fs1[i,i] = 0
            affinity_fs2[i,i] = 0
    
    elif method_id == 'euclidean':
        #=====================CASE 4===================================
        #               Euclidean distance kernel
        #==============================================================
        # euclidean_distances based kernels
        affinity_fs0 = -euclidean_distances(data,data)
        affinity_fs1 = -euclidean_distances(data_fs1,data_fs1)
        affinity_fs2 = -euclidean_distances(data_fs2,data_fs2)
    else:
        print "Error input for method_id"
    
    # spectral clustering result:
    clf_0 = SpectralClustering(n_clusters=K,affinity='precomputed')
    clf_0.fit(affinity_fs0)
    labels_predict = clf_0.labels_
    clf_1 = SpectralClustering(n_clusters=K,affinity='precomputed')
    clf_1.fit(affinity_fs1)
    labels_fs1_predict = clf_1.labels_
    clf_2 = SpectralClustering(n_clusters=K,affinity='precomputed')
    clf_2.fit(affinity_fs2)
    labels_fs2_predict = clf_2.labels_


plt.figure(1)
draw_similarity_matrix(affinity_fs0,labels_predict,K)
plt.figure(2)
draw_similarity_matrix(affinity_fs1,labels_fs1_predict,K)
plt.figure(3)
draw_similarity_matrix(affinity_fs2,labels_fs2_predict,K)

nmi_0 = normalized_mutual_info_score(gold,labels_predict)
nmi_1 = normalized_mutual_info_score(gold,labels_fs1_predict)
nmi_2 = normalized_mutual_info_score(gold,labels_fs2_predict)


#nmi_0 = nmi_revised(gold,labels_predict)
#nmi_1 = nmi_revised(gold,labels_fs1_predict)
#nmi_2 = nmi_revised(gold,labels_fs2_predict)


print nmi_0,nmi_1,nmi_2


#PCA
kpca_num = 3
gamma = 1.0/(2*sigma_rbf**2)
degree = 3


kpca_0 = KernelPCA(n_components=kpca_num,kernel='rbf',gamma=gamma,degree=degree)
kpca_0.fit(data)
kpca_data = kpca_0.fit_transform(data)
fig = plt.figure(4)
if kpca_num == 2:
    plt.scatter(kpca_data[:,0],kpca_data[:,1],c='b',marker='o')
else:
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(kpca_data[:,0],kpca_data[:,1],kpca_data[:,2],c='b')

kpca_1 = KernelPCA(n_components=kpca_num,kernel='rbf',gamma=gamma,degree=degree)
kpca_1.fit(data_fs1)
kpca_data_fs1 = kpca_1.transform(data_fs1)
fig = plt.figure(5)
if kpca_num == 2:
    plt.scatter(kpca_data_fs1[:,0],kpca_data_fs1[:,1],c='b',marker='o')
else:
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(kpca_data_fs1[:,0],kpca_data_fs1[:,1],kpca_data_fs1[:,2],c='b')

kpca_2 = KernelPCA(n_components=kpca_num,kernel='rbf',gamma=gamma,degree=degree)
kpca_2.fit(data_fs2)
kpca_data_fs2 = kpca_2.transform(data_fs2)
fig = plt.figure(6)
if kpca_num == 2:
    plt.scatter(kpca_data_fs2[:,0],kpca_data_fs2[:,1],c='b',marker='o')
else:
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(kpca_data_fs2[:,0],kpca_data_fs2[:,1],kpca_data_fs2[:,2],c='b')
"""
plt.show()

