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
file_data_train = open("/home/changyale/dataset/COPDGene/data_"+\
        "train_continuous.pkl","rb")
data_con_use,features_name_use = pickle.load(file_data_train)

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

# feature set<Add supervsion by including certain features>
fs = []
for i in range(len(features_name_use)):
    if features_name_use[i] in ['FEV1pp_utah','FEV1_FVC_utah']:
        fs.append(i)

# Choose dataset to use according to feature set
data_use = data[:,fs]

# Choose method to use: 'gmm' or 'dpgmm'
method = 'gmm'

if method == 'gmm':
    # Apply GMM and BIC to automatically find the number of clusters
    # The number of clusters should be no more than 10
    n_components_range = range(1,11)
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
        print "#Clusters",n_components,"BIC",bic[-1];
        print "lables_dist",labels_dist;
        print "====================================="
    clf = best_gmm
    for i in range(len(bic)):
        if bic[i] == min(bic):
            print "Found #Clusters",i+1;

if method == 'dpgmm':
    alpha_range = [1e-6,0.01,0.1,1.,10.,100.]
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

