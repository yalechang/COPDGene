"""This script applies backward search to do feature selection after removing
redundancy between features. This backward search should apply to continuous
and discrete features separately.
"""

#print __doc__

import pickle
import numpy as np
from time import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

t0 = time()
# Load training set
file_data_train = open("/home/changyale/dataset/COPDGene/data_train.pkl","rb")
info_con,info_dis,gold = pickle.load(file_data_train)
file_data_train.close()
data_con, features_name_con, features_type_con = info_con
data_dis, features_name_dis, features_type_dis = info_dis

# Load features id after removing redundancy
file_sel = open("data/features_sel_rm_redundancy.pkl","rb")
features_sel_con, features_name_con_1, features_sel_dis,features_name_dis_1 = \
        pickle.load(file_sel)
file_sel.close()

## Step 1: Prepare continuous data according to selected features
# Assert continuous features from two sources are the same 
assert set(features_name_con) == set(features_name_con_1)

# Choose dataset to use according to features_sel_con
data_con_use = np.zeros((data_con.shape[0],len(features_sel_con)))
for j in range(len(features_sel_con)):
    data_con_use[:,j] = data_con[:,features_sel_con[j]]

## Step 2: Prepare discrete data according to selected features
data_dis_1 = np.zeros((data_dis.shape[0],len(features_name_dis_1)))
index = 0
for j in range(len(features_name_dis)):
    if features_type_dis[j] == 'binary':
        data_dis_1[:,index] = data_dis[:,j]
        index += 1
assert index == len(features_name_dis_1)

#############################################################################
# NOTE that among these features 122 binary features, 27 features have values
# {0,1,3(unkown)}, 1 feature has values {1,2}. For the case of {0,1,3} valued
# features, how to tackle them?  
data_dis_use = np.zeros((data_dis_1.shape[0],len(features_sel_dis)))
for j in range(len(features_sel_dis)):
    data_dis_use[:,j] = data_dis_1[:,features_sel_dis[j]]
t1 = time()
print(["Preparing data takes "+str(t1-t0)+" seconds"])

## Backward search for continuous features
# Normalization of the continuous dataset
data = scale(data_con_use)
n_instances,n_features = data.shape

# Start with the full feature set 
bfs = range(n_features)
fs = [0]*n_features


while(len(bfs)>0):

