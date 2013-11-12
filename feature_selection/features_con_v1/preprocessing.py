"""
Parameters
----------
dataset_complete_knn.pkl: containing all the information about the dataset
after imputation, including the following:
    1) dataset: array, shape(8760, 211)
    2) features_name: list, len(211)
    3) features_type: list, len(211)

Returns
-------
data_train.pkl: containing all the information about the training
dataset(RandomGroupCode between 1 and 5):
    1) info_con: all the information of continuous features in the training set
        1.1) data_con: array, shape(4413, 76)
        1.2) features_name_con: list, len(76)
        1.3) features_type_con: list, len(76)
    2) info_dis: all the information of discrete features in the training set
        2.1) data_dis: array, shape(4413, 134)
        2.2) features_name_dis: list, len(134)
        2.3) features_type_dis: list, len(134)
    3) gold: list, len(4413)
        gold value for all the samples in the training set
"""

print __doc__

import numpy as np
import pickle

# Read into dataset
file_dataset = open("/home/changyale/dataset/COPDGene/dataset_complete_knn.pkl","rb")
dataset,features_name,features_type = pickle.load(file_dataset)
file_dataset.close()
n_instances,n_features = dataset.shape

# First extract 'RandomGroupCode'. Then use samples with RandomGroupCode 1-5 as
# training set, use samples with RandomGroupCode 6-10 as testing set
for j in range(n_features):
    if features_name[j] == 'RandomGroupCode':
        index_rgc = j
assert index_rgc<n_features

data_train = []
data_test = []

for i in range(n_instances):
    if float(dataset[i,index_rgc]) >=1 and float(dataset[i,index_rgc]) <= 5:
        data_train.append(list(dataset[i,0:index_rgc])+\
                [dataset[i,n_features-1]])
    else:
        data_test.append(list(dataset[i,0:index_rgc])+\
                [dataset[i,n_features-1]])

data_train = np.array(data_train)
data_test = np.array(data_test)

# Delete 'RandomGroupCode' related information in features information
del features_name[index_rgc]
del features_type[index_rgc]

##############################################################################
# In the following, we use data_train.
##############################################################################
data = data_train
n_instances,n_features = data.shape

# Separate data_test into continuous and discrete
n_features_con = 0
n_features_dis = 0
for j in range(len(features_type)):
    if features_type[j] in ['continuous','interval']:
        n_features_con += 1
    if features_type[j] in ['binary','categorical','ordinal']:
        n_features_dis += 1

data_con = np.zeros((n_instances,n_features_con))
data_dis = np.empty((n_instances,n_features_dis),dtype=list)
features_name_con = []
features_type_con = []
features_name_dis = []
features_type_dis = []
index_con = 0
index_dis = 0
for j in range(n_features):
    if features_type[j] in ['continuous','interval']:
        for i in range(n_instances):
            data_con[i,index_con] = float(data[i,j])
        index_con += 1
        features_name_con.append(features_name[j])
        features_type_con.append(features_type[j])
    if features_type[j] in ['binary','categorical','ordinal']:
        for i in range(n_instances):
            data_dis[i,index_dis] = data[i,j]
        index_dis += 1
        features_name_dis.append(features_name[j])
        features_type_dis.append(features_type[j])

# Obtain GOLD from 'FEV1pp_utah' and 'FEV1_FVC_utah'
gold = [0]*n_instances
for j in range(n_features_con):
    if features_name_con[j] == 'FEV1pp_utah':
        index_fev1 = j
    if features_name_con[j] == 'FEV1_FVC_utah':
        index_fvc = j
assert index_fev1<n_features_con and index_fvc<n_features_con

for i in range(n_instances):
    if data_con[i,index_fev1]>80 and data_con[i,index_fvc]>0.7:
        gold[i] = 0
    if data_con[i,index_fev1]>80 and data_con[i,index_fvc]<0.7:
        gold[i] = 1
    if data_con[i,index_fev1]>50 and data_con[i,index_fev1]<80 and \
            data_con[i,index_fvc]<0.7:
        gold[i] = 2
    if data_con[i,index_fev1]>30 and data_con[i,index_fev1]<50 and \
            data_con[i,index_fvc]<0.7:
        gold[i] = 3
    if data_con[i,index_fev1]<30 and data_con[i,index_fvc]<0.7:
        gold[i] = 4
    if data_con[i,index_fev1]<80 and data_con[i,index_fvc]>0.7:
        gold[i] = 5


info_con = [data_con,features_name_con,features_type_con]
info_dis = [data_dis,features_name_dis,features_type_dis]

file_result = open("/home/changyale/dataset/COPDGene/data_train.pkl","wb")
pickle.dump([info_con,info_dis,gold],file_result)
file_result.close()


