"""This script will apply spectral clustering on features with Normalized 
HISC matrix obtained by running "test_computing_HSIC.m". 
INPUT
-----
mtr_hsic.csv: HSIC values between pairwise features.

OUTPUT
------
Similarity matrix between features
"""
print __doc__

import numpy as np
import csv
import pickle
from sklearn.cluster.spectral import spectral_clustering
from sklearn.cluster import SpectralClustering
from python.COPDGene.utils.draw_similarity_matrix import draw_similarity_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_kernels
import copy
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA,KernelPCA

# Load HSIC matrix from csv file
file_hsic = open("mtr_hsic.csv","rb")
csvreader = csv.reader(file_hsic)
lines = [line for line in csvreader]
assert len(lines) == len(lines[0])
mtr_hsic = np.zeros((len(lines),len(lines[0])))
for i in range(mtr_hsic.shape[0]):
    for j in range(mtr_hsic.shape[1]):
        mtr_hsic[i,j] = np.float(lines[i][j])
file_hsic.close()

# Load Dataset from pickle file
# Load training set
file_data_train = open("/home/changyale/dataset/COPDGene/data_"+\
        "train_continuous.pkl","rb")
data_con_use,features_name_use = pickle.load(file_data_train)
file_data_train.close()

# Normalization of dataset
data = scale(data_con_use)

# Obtain Normalized HSIC matrix from HISC matrix
mtr_nhsic = np.zeros(mtr_hsic.shape)
for i in range(mtr_nhsic.shape[0]):
    for j in range(mtr_nhsic.shape[1]):
        mtr_nhsic[i,j] = mtr_hsic[i,j]/np.sqrt((mtr_hsic[i,i]*mtr_hsic[j,j]))

# Apply spectral clustering on the Normalized HSIC matrix
# Set the number of clusters
n_clusters_f = 5
labels_f = spectral_clustering(mtr_hsic,n_clusters=n_clusters_f,n_init=10)
cnt = [0]*n_clusters_f

tp = [[],[],[],[],[]]
tp_id = [[],[],[],[],[]]

for i in range(len(labels_f)):
    cnt[labels_f[i]] += 1
    tp[labels_f[i]].append(features_name_use[i])
    tp_id[labels_f[i]].append(i)
#print cnt
#print tp_id
#ax,pos_old = draw_similarity_matrix(mtr_nhsic,labels_f,n_clusters_f)
#plt.show()

flag_id = 0
for i in range(len(labels_f)):
    if features_name_use[i] == 'FVCpp_utah':
        flag_id = i
assert len(tp[labels_f[flag_id]]) == 12

# Feature selection results of backward search
fs_backward = [55,54,26,27,32,53,19,17,33,25,18,28,50]
fs_forward_supervision_1 = [46,48,47,52,53,49,35,58,25,56,54]
fs_forward_supervision_2 = [18,46,21,22,42,20,26,19,27]

sigma_rbf = 3.0
K = 6
#data_use = data[:,tp_id[labels_f[flag_id]]]
data_use = data[:,fs_forward_supervision_2]
#data_use = data
n_instances,n_features = data_use.shape

affinity = pairwise_kernels(data_use,data_use,metric='rbf',gamma=1./(2*sigma_rbf**2))

# Laplacian matrix
#mtr_w = copy.deepcopy(affinity_fs2)
#mtr_d = [0]*n_instances
#for i in range(n_instances):
#    mtr_w[i,i] = 0
#    mtr_d[i] = np.sum(mtr_w[i,:])
#mtr_l = np.zeros((n_instances,n_instances))
#for i in range(n_instances):
#    for j in range(i,n_instances):
#        mtr_l[i,j] = mtr_w[i,j]/np.sqrt(mtr_d[i]*mtr_d[j])
#        mtr_l[j,i] = mtr_l[i,j]
#eig_val, eig_vec = np.linalg.eig(mtr_l)

clf = SpectralClustering(n_clusters=K,affinity='precomputed')
clf.fit(affinity)
labels_predict = clf.labels_

draw_similarity_matrix(affinity,labels_predict,K)

#PCA
gamma = 1.0/(2*sigma_rbf**2)
degree = 3
color = ['b','r','g','m','y','k']

kpca_2 = KernelPCA(n_components=2,kernel='rbf',gamma=gamma,degree=degree)
kpca_2.fit(data_use)
kpca_2_data = kpca_2.fit_transform(data_use)
fig = plt.figure(1)
for i in range(len(labels_predict)):
    plt.scatter(kpca_2_data[i,0],kpca_2_data[i,1],c=color[labels_predict[i]],\
            marker='o')

kpca_3 = KernelPCA(n_components=3,kernel='rbf',gamma=gamma,degree=degree)
kpca_3.fit(data_use)
kpca_3_data = kpca_3.fit_transform(data_use)
fig = plt.figure(2)
ax = fig.add_subplot(111,projection='3d')
for i in range(len(labels_predict)):
    ax.scatter(kpca_3_data[i,0],kpca_3_data[i,1],kpca_3_data[i,2],\
            c=color[labels_predict[i]],marker='o')

plt.show()

