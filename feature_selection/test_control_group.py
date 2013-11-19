""" This script test whether control group samples are clustered in the same
group
"""
print __doc__

import pickle
import numpy as np
from time import time
from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from python.COPDGene.utils.sample_wr import sample_wr

# Number of clusters
K = 4

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

# Obtain gold value
gold = [0]*n_instances
# Get the feature index of 'FEV1pp_utah' and 'FEV1_FVC_utah'
index_FEV1pp_utah = 0
index_FEV1_FVC_utah = 0
for j in range(len(features_name_use)):
    if features_name_use[j] == 'FEV1pp_utah':
        index_FEV1pp_utah = j
    if features_name_use[j] == 'FEV1_FVC_utah':
        index_FEV1_FVC_utah = j

for i in range(n_instances):
    if data_con_use[i,index_FEV1pp_utah]>80 and data_con_use[i,index_FEV1_FVC_utah]>0.7:
        gold[i] = 0
    if data_con_use[i,index_FEV1pp_utah]>80 and data_con_use[i,index_FEV1_FVC_utah]<0.7:
        gold[i] = 1
    if data_con_use[i,index_FEV1pp_utah]>50 and data_con_use[i,index_FEV1pp_utah]<80 and \
            data_con_use[i,index_FEV1_FVC_utah]<0.7:
        gold[i] = 2
    if data_con_use[i,index_FEV1pp_utah]>30 and data_con_use[i,index_FEV1pp_utah]<50 and \
            data_con_use[i,index_FEV1_FVC_utah]<0.7:
        gold[i] = 3
    if data_con_use[i,index_FEV1pp_utah]<30 and data_con_use[i,index_FEV1_FVC_utah]<0.7:
        gold[i] = 4
    if data_con_use[i,index_FEV1pp_utah]<80 and data_con_use[i,index_FEV1_FVC_utah]>0.7:
        gold[i] = 5

# final selected feature set after forward search
#fs = [1,40,9,7,26,31,35,8,51,38]
# final selected feature set after backward search
#fs = [58,54,59,25,27,32,23,24,26,30,55,31,33,37,61,60]
# final selected feature set after forward search with supervision 1
#fs = [51,53,52,57,63,40,58,54,61,33,59]
# final selected feature set after forward search with supervision 2
fs = [23,51,26,27,10,25,38,8,24,31,32]

# Choose dataset to use according to feature set
data_use = data[:,fs]

# Clustering using KMeans
score = []
for i in range(50):
    print "============================"
    print i
    estimator_1 = KMeans(init='random',n_clusters=K,n_init=10,n_jobs=-1)
    estimator_1.fit(data_use)
    data_use_labels = estimator_1.labels_
    score.append(estimator_1.inertia_)
    print score[-1]
    counter = np.arange(K*6).reshape(K,6)
    for i in range(K):
        for j in range(6):
            counter[i,j] = 0
    for i in range(n_instances):
        counter[data_use_labels[i],gold[i]] += 1
    print counter
    print sum(counter.T)
for i in range(50):
    if score[i] == min(score):
        print i


