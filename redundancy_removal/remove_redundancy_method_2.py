"""This method ranks features and then removes redundancy
"""
print __doc__

import numpy as np
import pickle
import csv
import networkx as nx
import matplotlib.pyplot as plt
import copy
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import random
from sklearn.metrics import normalized_mutual_info_score
from time import time
from my_hierarchical_clustering import my_hierarchical_clustering
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
import copy

# Load training dataset
file_data_train = open("data_train.pkl","rb")
info_con_train,info_dis_train,gold_train = pickle.load(file_data_train)
data_con_train, features_name_con_train, features_type_con_train = \
        info_con_train
file_data_train.close()

# Load testing dataset
file_data_test = open("data_test.pkl","rb")
info_con_test,info_dis_test,gold_test = pickle.load(file_data_test)
data_con_test, features_name_con_test, features_type_con_test = info_con_test
file_data_test.close()

# Load affinity matrix
file_hsic = open("mtr_hsic_nhsic.pkl","rb")
mtr_hsic, mtr_nhsic = pickle.load(file_hsic)
file_hsic.close()

# Load features ranking information
file_ranking = open("exp_result/ranking_con_Ref2.csv","rb")
reader = csv.reader(file_ranking)
lines = [line for line in reader]
features_ordered = {lines[i][3]:i-1 for i in range(1,len(lines))}
file_ranking.close()

# Obtain the features rank using feature names
features_rank = range(len(features_ordered))
features_name = features_name_con_train
for i in range(len(features_name)):
    for key in features_ordered:
        if features_name[i] == key:
            features_rank[i] = features_ordered[key]

# Obtain the number of continuous features
n_features = mtr_nhsic.shape[0]

thd = 0.7
for i in range(n_features-1):
    for j in range(i+1,n_features):
        if mtr_nhsic[i,j]>thd:
            #print i,j
            pass

##############################################################################
# Method 2: Rank features and then remove redundancy
##############################################################################
# Order features according to ranking
features_ranked = range(n_features)
for i in range(n_features):
    features_ranked[features_rank[i]] = i 

# Select features
features_sel = [features_ranked[0]]
for i in range(1,n_features):
    flag = True
    for j in range(len(features_sel)):
        if mtr_nhsic[features_ranked[i],features_sel[j]] > thd:
            flag = False
            break
    if flag == True:
        features_sel.append(features_ranked[i])

mtr_sim = np.zeros((len(features_sel),len(features_sel)))
ii = 0
jj = 0
labels = []
for i in range(n_features):
    if i in features_sel:
        jj = 0
        for j in range(n_features):
            if j in features_sel:
                mtr_sim[ii,jj] = mtr_nhsic[i,j]
                jj += 1
        ii += 1
        labels.append(features_name[i])
# Draw the dendrogram according to linkage matrix
fig = plt.figure(figsize=(20,20))
mtr_lin = my_hierarchical_clustering(mtr_sim,method='complete')
tmp = copy.deepcopy(mtr_lin)
for i in range(tmp.shape[0]):
    tmp[i,2] = 1-tmp[i,2]
dend = dendrogram(tmp,labels=labels)
plt.savefig('dendrogram_method_2.png')

file_remove_redundancy = open("remove_redundancy_method_2.csv","wb")
file_writer = csv.writer(file_remove_redundancy)
file_writer.writerow(["Selected Features"])
file_writer.writerow(["Feature ID","Feature Name","Feature Rank"])

for i in range(n_features):
    if i in features_sel:
        file_writer.writerow([i,features_name[i],features_rank[i]])

for i in range(3):
    file_writer.writerow([''])

file_writer.writerow(["Removed Features"])
file_writer.writerow(["Feature ID","Feature Name","Feature Rank"])

for i in range(n_features):
    if i not in features_sel:
        file_writer.writerow([i,features_name[i],features_rank[i]])

file_remove_redundancy.close()

