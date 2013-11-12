"""This script runs forward search for all the 'continuous'(not including
'interval', 'categorical', 'binary', 'ordinal') features. Note that here we
don't have to specify the number of clusters.                                   
"""

print __doc__

import pickle
import numpy as np
from time import time
#from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from python.COPDGene.utils.sample_wr import sample_wr

############################# Parameter Setting ##############################
# we don't need to set the number of clusters
##############################################################################

t0 = time()
# Load training set
file_data_train = open("/home/changyale/dataset/COPDGene/data_train.pkl","rb")
info_con,info_dis,gold = pickle.load(file_data_train)
file_data_train.close()
data_con, features_name_con, features_type_con = info_con

# Choose only 'continuous' features for backward search
data_con_use = []
features_name_use = []
for j in range(len(features_type_con)):
    if features_type_con[j] == 'continuous':
        data_con_use.append(data_con[:,j])
        features_name_use.append(features_name_con[j])
data_con_use = np.array(data_con_use).T

# Prepare reference dataset for continuous features
# Random sample with replacement from training set to form a reference dataset
data_con_use_ref = np.zeros((data_con_use.shape[0],data_con_use.shape[1]))
for j in range(data_con_use.shape[1]):
    tp_index = sample_wr(range(data_con_use_ref.shape[0]),\
            data_con_use_ref.shape[0])
    for i in range(len(tp_index)):
        data_con_use_ref[i,j] = data_con_use[tp_index[i],j]

t1 = time()
print(["Preparing data takes "+str(t1-t0)+" seconds"])

# Forward search for continuous features
# Normalization of the continuous dataset
data = scale(data_con_use)
#data = data_con_use
n_instances, n_features = data.shape

data_ref = scale(data_con_use_ref)

# feature set
fs = [21,4,6,8,5,16,9,22,17]

# Choose dataset to use according to feature set
data_use = data[:,fs]

# Choose method to use: 'gmm' or 'dpgmm'
method = 'gmm'

if method == 'gmm':
    # Apply GMM and BIC to automatically find the number of clusters
    n_components_range = range(1,20)
    bic = []
    logl = []
    lowest_bic = np.infty
    for n_components in n_components_range:
        # Fit a mixture of gaussians with EM
        gmm = mixture.GMM(n_components=n_components,covariance_type='full')
        gmm.fit(data_use)
        tmp = gmm.eval(data_use)
        logl.append(tmp[0].sum())
        labels_predict = gmm.predict(data_use)
        labels_unique = np.unique(labels_predict)
        # Counting the number of samples belonging to each cluster
        labels_dist = [0]*len(labels_unique)
        for i in range(len(labels_predict)):
            for j in range(len(labels_unique)):
                if labels_predict[i] == labels_unique[j]:
                    labels_dist[j] += 1
        bic.append(gmm.bic(data_use))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
        print n_components,"labels:",labels_unique,"lables_dist",labels_dist
    clf = best_gmm
    for i in range(len(bic)):
        print([i,bic[i],logl[i]])

if method == 'dpgmm':
    alpha_range = [0.001,0.01,0.1,1.,10.,100.,1000.,1e6]
    n_clusters_max = 100
    for alpha in alpha_range:
        # Fit a mixuture of gaussians with Dirichlet Process Mixture
        dpgmm = mixture.DPGMM(n_components=n_clusters_max,\
                covariance_type='full',alpha=alpha,n_iter=1000)
        dpgmm.fit(data_use)
        labels_predict = dpgmm.predict(data_use)
        labels_unique = np.unique(labels_predict)
        # Counting the number of samples belonging to each cluster
        labels_dist = [0]*len(labels_unique)
        for i in range(len(labels_predict)):
            for j in range(len(labels_unique)):
                if labels_predict[i] == labels_unique[j]:
                    labels_dist[j] += 1
        print(["alpha: ",alpha,"labels: ",labels_unique,"labels_dist: ",\
                labels_dist])
    print(["Upper bound for the number of clusters: ",n_clusters_max])


