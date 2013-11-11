"""This script runs forward search with entropy as metric for clustering
quality
"""
print __doc__

import pickle
import numpy as np
from time import time
from sklearn.preprocessing import scale
from python.COPDGene.utils.entropy_metric import entropy_metric_1

t0 = time()
# Load training set
file_data_train = open("/home/changyale/dataset/COPDGene/data_train.pkl","rb")
info_con,info_dis,gold = pickle.load(file_data_train)
file_data_train.close()
data_con, features_name_con, features_type_con = info_con

# Choose only 'continuous' features for forward search
data_con_use = []
features_name_use = []
for j in range(len(features_type_con)):
    if features_type_con[j] == 'continuous':
        data_con_use.append(data_con[:,j])
        features_name_use.append(features_name_con[j])
data_con_use = np.array(data_con_use).T

t1 = time()
print(["Preparing data takes "+str(t1-t0)+" seconds"])

# Forward search for continuous features
# Normalization of the continuous dataset
data = scale(data_con_use)
n_instances, n_features = data.shape

# Start with the empty feature set
bfs = []

# Use entropy as metric for clustering quality
score_best = [0]*n_features
features_add = []
score = range(n_features)
fs = range(n_features)

while(len(bfs)<=n_features):
    to = time()
    for i in range(n_features):
        t1 = time()
        if i in set(bfs):
            score[i] = np.infty
            fs[i] = bfs
        else:
            fs[i] = bfs+[i]
            data_use = data[:,fs[i]]
            score[i] = entropy_metric_1(data_use)
            t2 = time()
            print([i,t2-t1])
    for i in range(n_features):
        if score[i] == min(score):
            score_best[len(bfs)] = min(score)
            features_add.append(i)
            bfs = fs[i]
            break
    t3 = time()
    print([score_best[len(bfs)-1],features_add,"RunningTime(s): ",(t3-t0)])

