import pickle
import csv
import numpy as np
import copy

file_data = open("data_imputation_questionnaire.pkl","rb")
data_features,features_type = pickle.load(file_data)
file_data.close()

n_instances,n_features = data_features.shape
features_name = data_features[0,:]
data_whole = data_features[1:n_instances,:]
n_instances,n_features = data_whole.shape

file_data = open("info_missing_type_include.pickle","rb")
tp1,tp2,tp3,tp4,features_include = pickle.load(file_data)
file_data.close()

for i in range(len(tp1)):
    assert tp1[i] == features_name[i]

# Extract dataset by only choosing included features
features_name_include = []
features_type_include = []
data = np.empty((n_instances,sum(features_include)),dtype=list)
index = 0
for j in range(len(features_include)):
    if features_include[j] == True:
        for i in range(n_instances):
            data[i,index] = data_whole[i,j]
        index += 1
        features_name_include.append(features_name[j])
        features_type_include.append(features_type[j])

tp_features_name_include = copy.copy(features_name_include)

# Compute the percentage of missingness of included features
n_instances,n_features = data.shape
n_missing = 0
n_missing_none = 0
n_missing_all = 0

for i in range(n_instances):
    for j in range(n_features):
        if data[i,j] == '':
            n_missing += 1
        if data[i,j] == None:
            n_missing_none += 1
        if data[i,j] in ['',None]:
            n_missing_all += 1

print n_missing,n_missing_none,n_missing_all

percent = n_missing*1./(n_instances*n_features)
