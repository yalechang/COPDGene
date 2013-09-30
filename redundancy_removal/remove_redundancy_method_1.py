"""This script apply hierarchical clustering recursively to remove redundancy
between features
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
# Method 1: Apply hierarchical clustering recursively
##############################################################################
flag = True
mtr_sim = mtr_nhsic

# store the original id of features in corresponding locations
features_id_map = range(n_features)

# indicate whether a feature should be kept
flag_keep = [True]*n_features

n_iter = 0

while flag == True:
    print n_iter
    n_iter += 1
    
    # Apply hierarchical clustering
    mtr_lin = my_hierarchical_clustering(mtr_sim,method='complete')
     # Draw the dendrogram according to linkage matrix
    fig = plt.figure(figsize=(20,20))
    labels = []
    for i in features_id_map:
        labels.append(features_name[i])
    tmp = copy.deepcopy(mtr_lin)
    for i in range(tmp.shape[0]):
        tmp[i,2] = 1-tmp[i,2]
    dend = dendrogram(tmp,labels=labels)
    plt.savefig('dendrogram_method_1_iter_'+str(n_iter)+'.png')

    
    n_features = mtr_sim.shape[0]
    #print mtr_lin[0,:]
    # Find the cut location in the tree
    for i in range(n_features):
        if mtr_lin[i,2] < thd:
            cut_loc = i
            break
    
    # Determine if the hierarchical clustering process is continued
    # cut_loc == 0 means values in all nodes are less than the threshold 
    flag = cut_loc > 0
    
    if flag == False:
        break
    
    # Remove redundant nodes below the found location
    # Collection of features that are redundant
    features_redundant = []
    for i in range(cut_loc):
        if mtr_lin[i,0]<n_features and mtr_lin[i,1]<n_features:
            features_redundant.append([mtr_lin[i,0],mtr_lin[i,1]])
    
    # features_redundant_original_id
    features_roid = []
    for i in range(len(features_redundant)):
        features_roid.append([features_id_map[int(features_redundant[i][0])],\
                features_id_map[int(features_redundant[i][1])]])

    # Keep the feature with the higher rank
    for i in range(len(features_roid)):
        if features_rank[features_roid[i][0]] < \
                features_rank[features_roid[i][1]]:
            flag_keep[features_roid[i][1]] = False
        else:
            flag_keep[features_roid[i][0]] = False

    # Update the similarity matrix
    n_features = sum(flag_keep)
    mtr_sim = np.zeros((n_features,n_features))
    ii = 0
    jj = 0
    features_id_map = []
    for i in range(mtr_nhsic.shape[0]):
        if flag_keep[i] == True:
            jj = 0
            for j in range(mtr_nhsic.shape[1]):
                if flag_keep[j] == True:
                    mtr_sim[ii,jj] = mtr_nhsic[i,j]
                    jj += 1
            ii += 1
            features_id_map.append(i)

for i in range(len(flag_keep)):
    #print i,flag_keep[i],features_rank[i]
    if flag_keep[i] == False:
        print i

# Check to see if there are values larger than threshold in the final mtr_sim
flag_right = True
for i in range(mtr_sim.shape[0]-1):
    for j in range(i+1,mtr_sim.shape[1]):
        if mtr_sim[i,j] > thd:
            flag_right = False
print flag_right

file_remove_redundancy = open("remove_redundancy_method_1.csv","wb")
file_writer = csv.writer(file_remove_redundancy)
file_writer.writerow(["Selected Features"])
file_writer.writerow(["Feature ID","Feature Name","Feature Rank"])
for i in range(len(flag_keep)):
    if flag_keep[i] == True:
        file_writer.writerow([i,features_name[i],features_rank[i]])

for i in range(3):
    file_writer.writerow([''])

file_writer.writerow(["Removed Features"])
file_writer.writerow(["Feature ID","Feature Name","Feature Rank"])
for i in range(len(flag_keep)):
    if flag_keep[i] == False:
        file_writer.writerow([i,features_name[i],features_rank[i]])
file_remove_redundancy.close()
