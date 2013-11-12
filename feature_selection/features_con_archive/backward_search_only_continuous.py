"""This script runs backward for all the 'continuous'(not including 'interval'
'categorical','binary','ordinal') features. Note that the differences lie in
two aspects: 1, we only use continuous features;
             2, We don't apply redundancy removal before backward search. 
"""

print __doc__

import pickle
import numpy as np
from time import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from python.COPDGene.utils.sample_wr import sample_wr

############################# Parameter Setting ##############################
# The number of clusters in KMeans
K = 4
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

# Backward search for continuous features
# Normalization of the continuous dataset
data = scale(data_con_use)
n_instances, n_features = data.shape

data_ref = scale(data_con_use_ref)

# Start with the full feature set 
bfs = range(n_features)

# Silhouette score for original dataset and reference dataset using the same
# set of features
score_best = [0]*n_features
score_best_ref = [0]*n_features 
features_rm = []

while(len(bfs)>1):
    t0 = time()
    # Candidate feature sets, each element is a feature set.
    fs = [0]*len(bfs)
    # Scores(Clustering Metric Value) for each feature set
    score = [0]*len(bfs)
    for i in range(len(bfs)):
        t1 = time()
        # i-th candidate feature sets
        fs[i] = list(set(bfs)-set([bfs[i]]))
        # Prepare dataset using i-th candidate feature sets
        data_use = data[:,fs[i]]
        estimator = KMeans(init='random',n_clusters=K,n_init=10,n_jobs=-1)
        estimator.fit(data_use)
        data_use_labels = estimator.labels_
        score[i] = silhouette_score(data_use,data_use_labels,\
                metric='euclidean')
        t2 = time()
        print([i,t2-t1])
    for i in range(len(bfs)):
        if score[i] == max(score):
            score_best[len(bfs)-1] = max(score)
            # Compute Silhouette for reference data using the same set of
            # features
            data_ref_use = data_ref[:,fs[i]]
            estimator_0 = KMeans(init='random',n_clusters=K,n_init=10,n_jobs=-1)
            estimator_0.fit(data_ref_use)
            data_ref_use_labels = estimator_0.labels_
            score_best_ref[len(bfs)-1] = silhouette_score(data_ref_use,\
                    data_ref_use_labels,metric='euclidean')
            features_rm.append(bfs[i])
            bfs = fs[i]
            break
    t3 = time()
    print([score_best[len(bfs)],score_best_ref[len(bfs)],features_rm,\
            "RunningTime(s): ",(t3-t0)])


