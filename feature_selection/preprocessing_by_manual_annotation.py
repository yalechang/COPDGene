""" This script extract continuous features from imputated dataset according 
to feature annotation information presented in
"features_included_info_mhc20131104.txt". The difference of this preprocessing
step with the what's shown in "preprocessing.py" is that we use expert
annotation information instead of automatic annotation information to extract
continuous features. In this sense, the method here is more flexible for adding
manual supervision.

Parameters
----------
dataset_complete_knn.pkl: containing all the information about the dataset
after imputation, including the following:
    1) dataset: array, shape(8760, 211)
    2) features_name: list, len(211)
    3) features_type: list, len(211), Note that these feature types are
    automatically annotated by the algorithm. But it might not be accurate
    because of ignorance of physical meaning of the features. So we should look
    at the information in expert annotation as the ground truth.

features_included_info_mhc20131104.txt: containing annotations of all the
continuous features. Determine whether to select a feature according to the
manual annotation information presented in this file. 

Returns
-------
data_train_continuous.pkl: containing all the information about continuous 
features of the training set(RandomGroupCode between 1 and 5):
    1) data_train_con: array, shape(4413,n_features_con)
    2) features_name_con: list, len(n_features_con)
"""

print __doc__

import numpy as np
import pickle
from python.COPDGene.utils.read_txt_data import read_txt_data

# Load imputed dataset
file_dataset = open("/home/changyale/dataset/COPDGene/dataset_complete_knn_dropoutlier_11-4-13.pkl","rb")
dataset,features_name,features_type = pickle.load(file_dataset)
file_dataset.close()
n_instances,n_features = dataset.shape

# Find the feature index of 'RandomGroupCode'. Then use samples with
# RandomGroupCode 1-5 as training set
for j in range(n_features):
    if features_name[j] == 'RandomGroupCode':
        index_rgc = j
assert index_rgc<n_features

data_train = []
for i in range(n_instances):
    if float(dataset[i,index_rgc]) >=1 and float(dataset[i,index_rgc]) <= 5:
        data_train.append(list(dataset[i,:]))
data_train = np.array(data_train)

# Extract continuous features from data_train according to expert annotation
file_name = "/home/changyale/dataset/COPDGene/features_included_info_"+\
        "mhc20131104_with_times.txt"
features_annotation = read_txt_data(file_name)
features_name_1 = features_annotation[1:features_annotation.shape[0],0]
for i in range(len(features_name)):
    assert features_name[i] == features_name_1[i].replace("\"","")


features_id_con = []
features_name_con = []
for i in range(len(features_name)):
    if features_annotation[i+1,3] != "\"Y\"":
        features_id_con.append(i)
        features_name_con.append(features_name[i])
#for i in range(len(features_id_con)):
#    print features_id_con[i],features_name_con[i]

data_train_con = np.double(data_train[:,features_id_con])

# Save the continuous features to pickle and csv file
file_name = "/home/changyale/dataset/COPDGene/data_train_continuous.pkl"
file_result = open(file_name,"wb")
pickle.dump([data_train_con,features_name_con],file_result)
file_result.close()

