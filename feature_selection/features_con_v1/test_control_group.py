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

# Obtain gold value
gold = [0]*n_instances
for i in range(n_instances):
    if data_con_use[i,35]>80 and data_con_use[i,37]>0.7:
        gold[i] = 0
    if data_con_use[i,35]>80 and data_con_use[i,37]<0.7:
        gold[i] = 1
    if data_con_use[i,35]>50 and data_con_use[i,35]<80 and \
            data_con_use[i,37]<0.7:
        gold[i] = 2
    if data_con_use[i,35]>30 and data_con_use[i,35]<50 and \
            data_con_use[i,37]<0.7:
        gold[i] = 3
    if data_con_use[i,35]<30 and data_con_use[i,37]<0.7:
        gold[i] = 4
    if data_con_use[i,35]<80 and data_con_use[i,37]>0.7:
        gold[i] = 5

# final selected feature set after forward search
#fs = [21,4,6,8,5,16,9,22,17]
# final selected feature set after backward search
#fs = [29,30,21,9,8,16,17,22]
# final selected feature set after forward search with supervision
fs = [35,37,36,45,24,41,47,42,18]

# Choose dataset to use according to feature set
data_use = data[:,fs]

# Clustering using KMeans
estimator_1 = KMeans(init='random',n_clusters=K,n_init=10,n_jobs=-1)
estimator_1.fit(data_use)
data_use_labels = estimator_1.labels_

counter = np.zeros((K,6))
for i in range(n_instances):
    counter[data_use_labels[i],gold[i]] += 1

print counter
