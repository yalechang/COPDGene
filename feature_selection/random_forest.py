"""This script implements the following functions:
    1) Rank all the features(including continuous and discrete) using random
    forest.
    2) Remove redundancy between features according the pairwise Normalized
    HSIC.
    3) Save the ids of selected features for continuous and discrete
    respectively.
"""

print __doc__

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from python.COPDGene.utils.sample_wr import sample_wr
from time import time
from python.COPDGene.utils.compute_similarity_from_leaf_index import \
        compute_similarity_from_leaf_index
import pickle
from sklearn.cluster import spectral_clustering
import copy
import csv
from python.COPDGene.utils.remove_redundancy import remove_redundancy
from python.COPDGene.utils.obtain_rank_use_score import obtain_rank_use_score

########################### Parameter Settings################################
# Number of clusters of samples for applying spectral clustering
n_clusters = 4
# Threshold for removing redundancy
thd = 0.7
##############################################################################


# Load complete dataset
file_dataset_complete = open("/home/changyale/dataset/COPDGene/dataset_complete_mean.pkl","rb")

# Extract dataset values, features_name and features_type
dataset,features_name,features_type = pickle.load(file_dataset_complete)
file_dataset_complete.close()

# Preprocessing of the dataset
n_binary = 0
n_categorical = 0
n_continuous = 0
n_interval = 0
n_ordinal = 0

# Store dataset that only contains 'binary', 'continuous' and 'interval'
data_use = []
features_name_use = []
features_type_use = []
for j in range(len(features_type)):
    if features_type[j] == "binary":
        n_binary += 1
        data_use.append(list(np.double(dataset[:,j])))
        features_name_use.append(features_name[j])
        features_type_use.append(features_type[j])
    elif features_type[j] == "categorical":
        n_categorical += 1
        #print "categorical",j,features_name[j]
    elif features_type[j] == "continuous":
        n_continuous += 1
        data_use.append(list(np.double(dataset[:,j])))
        features_name_use.append(features_name[j])
        features_type_use.append(features_type[j])
    elif features_type[j] == "interval":
        n_interval += 1
        data_use.append(list(np.double(dataset[:,j])))
        features_name_use.append(features_name[j])
        features_type_use.append(features_type[j]) 
    elif features_type[j] == "ordinal":
        n_ordinal += 1
        #print "ordinal",j,features_name[j]
    else:
        pass
#print n_binary,n_categorical,n_continuous,n_interval,n_ordinal
data_use = np.array(data_use).T
data = data_use

# Separate training set and testing set using RandomGroupCode
for j in range(len(features_name_use)):
    if features_name_use[j] == "RandomGroupCode":
        index_RandomGroupCode = j
data_train = []
data_test = []
for i in range(data.shape[0]):
    if data[i,index_RandomGroupCode]>=0 and data[i,index_RandomGroupCode]<=5:
        data_train.append(list(data[i,0:index_RandomGroupCode])+\
                list(data[i,index_RandomGroupCode+1:data.shape[1]]))
    if data[i,index_RandomGroupCode]>=6 and data[i,index_RandomGroupCode]<=10:
        data_test.append(list(data[i,0:index_RandomGroupCode])+\
                list(data[i,index_RandomGroupCode+1:data.shape[1]]))
data_train = np.array(data_train)
data_test = np.array(data_test)

# Remove "RandomGroupCode" in the feature information accordingly
del features_name_use[index_RandomGroupCode]
del features_type_use[index_RandomGroupCode]

# Load HSIC matrix for continuous features and discrete features
file_hsic = open("data/mtr_hsic_nhsic_con.pkl","rb")
mtr_hsic_con,mtr_nhsic_con = pickle.load(file_hsic)
file_hsic.close()
file_hsic = open("data/mtr_hsic_nhsic_dis.pkl","rb")
mtr_hsic_dis,mtr_nhsic_dis = pickle.load(file_hsic)
file_hsic.close()

# Load information about continuous and discrete features
file_data_train = open("/home/changyale/dataset/COPDGene/data_train.pkl","rb")
info_con,info_dis,gold = pickle.load(file_data_train)
file_data_train.close()
data_con, features_name_con, features_type_con = info_con
data_dis, features_name_dis, features_type_dis = info_dis

# Random sample with replacement from data_train to form a reference dataset
data_train_ref = np.zeros((data_train.shape[0],data_train.shape[1]))
for j in range(data_train_ref.shape[1]):
    tp_index = sample_wr(range(data_train_ref.shape[0]),data_train_ref.shape[0])
    for i in range(len(tp_index)):
        data_train_ref[i,j] = data_train[tp_index[i],j]

# Label data_train as class 0 and data_train_ref as class 1, resulting in a
# dataset "data_use" and its label "labels"
labels = []
data_use = np.zeros((data_train.shape[0]+data_train_ref.shape[0],\
        data_train.shape[1]))
for i in range(data_train.shape[0]):
    data_use[i,:] = data_train[i,:]
    labels.append(0)
for i in range(data_train_ref.shape[0]):
    data_use[data_train.shape[0]+i,:] = data_train_ref[i,:]
    labels.append(1)
labels = np.array(labels)

# Apply Random Forest Classifier on the dataset to obtain importance score
t0 = time()
clf = RandomForestClassifier(n_estimators=1000,criterion='gini',max_depth=None,\
        min_samples_split=2,min_samples_leaf=1,max_features='auto',\
        bootstrap=True,oob_score=True,n_jobs=-1,random_state=None,\
        verbose=0,compute_importances=None)
clf.fit(data_use,labels)
t1 = time()
print(["Random Forest Running Time(min):",(t1-t0)/60])

# Compute similarity matrix for the training set
#leaf_index = clf.apply(data_train)
#t0 = time()
#mtr_affinity = compute_similarity_from_leaf_index(np.double(leaf_index))
#t1 = time()
#print(["Similarity matrix computation time(min)",(t1-t0)/60])

# Store the similarity matrix for future use
#file_result = open("mtr_affinity_random_forest.pkl","wb")
#pickle.dump(mtr_affinity,file_result)
#file_result.close()

# Apply spectral clustering given the similarity matrix as input. Note here
# since we already have similarity matrix, we don't have to specify the scale
# parameter to construct the kernel matrix
#t0 = time()
#labels_train = spectral_clustering(mtr_affinity,n_clusters=n_clusters,\
#        n_components=None,eigen_solver=None,random_state=None,n_init=10,\
#        k=None,eigen_tol=0.0,assign_labels='kmeans',mode=None)
#t1 = time()
#print(["Spectral Clustering time(min)",(t1-t0)/60])

# (Optional)Get the eigenvalues associated with the similarity matrix

features_importance = clf.feature_importances_
tp = copy.copy(features_name_use)

# Rank features according to features_importance
features_rank_con = range(n_continuous+n_interval)
features_rank_dis = range(n_binary)

# Separate importance score into continuous and discrete
# To distinguish the features name and type obtained from data_train.pkl, we
# add '1' after the ending position.
features_importance_con = []
features_name_con_1 = []
features_type_con_1 = []

features_importance_dis = []
features_name_dis_1 = []
features_type_dis_1 = []

for i in range(len(features_importance)):
    if features_type_use[i] in ['continuous','interval']:
        features_importance_con.append(features_importance[i])
        features_name_con_1.append(features_name_use[i])
        features_type_con_1.append(features_type_use[i])
    if features_type_use[i] == 'binary':
        features_importance_dis.append(features_importance[i])
        features_name_dis_1.append(features_name_use[i])
        features_type_dis_1.append(features_type_use[i])
assert len(features_importance_con) == n_continuous+n_interval-1
assert len(features_importance_dis) == n_binary

# Check whether continuous features info obtained from data_train.pkl and
# data_complete_mean.pkl are the same(Expected to be the same)
# Then we can directly use mtr_nhsic_con as input for redundancy removal
assert len(features_name_con) == len(features_name_con_1)
flag = True
for i in range(len(features_name_con)):
    if features_name_con[i] != features_name_con_1[i]:
        flag = False
        break
assert flag == True

# Remove redundancy for continuous features
features_rank_con, features_importance_con_ranked = \
        obtain_rank_use_score(features_name_con_1,features_importance_con)
features_sel_con = remove_redundancy(features_rank_con,thd,mtr_nhsic_con)

# Since we only keep binary features, features in features_name_dis_1 should be
# a subset of features_name_dis
index_keep = []
for i in range(len(features_name_dis)):
    if features_name_dis[i] in features_name_dis_1:
        index_keep.append(i)
mtr_nhsic_dis_1 = np.zeros((len(index_keep),len(index_keep)))
for i in range(len(index_keep)):
    for j in range(len(index_keep)):
        mtr_nhsic_dis_1[i,j] = mtr_nhsic_dis[index_keep[i],index_keep[j]]

# Remove redundancy for discrete features
features_rank_dis, features_importance_dis_ranked = \
        obtain_rank_use_score(features_name_dis_1,features_importance_dis)
features_sel_dis = remove_redundancy(features_rank_dis,thd,mtr_nhsic_dis_1)

# Save the selected features
file_result = open("data/features_sel_rm_redundancy.pkl","wb")
result = [features_sel_con,features_name_con_1,\
        features_sel_dis,features_name_dis_1]
pickle.dump(result,file_result)
file_result.close()

# Test the stability of redundancy removal
print features_sel_con[0:20]
print features_sel_dis[0:20]

# Write the ranked features into a csv file
file_result = open("data/features_importance.csv","wb")
file_writer = csv.writer(file_result)
file_writer.writerow(["Rank","Feature Name","Feature Score"])
for i in range(len(tp)):
    file_writer.writerow([i+1,tp[i],features_importance[i]])
file_result.close()

# Construct Laplacian from similarity matrix
# Degree matrix
#mtr_d = sum(mtr_affinity)

# Laplacian matrix
#n_instances = mtr_affinity.shape[0]
#mtr_l = np.zeros((n_instances,n_instances))
#for i in range(n_instances):
#    for j in range(n_instances):
#        mtr_l[i,j] = 1./np.sqrt(mtr_d[i])*mtr_affinity[i,j]*1./np.sqrt(mtr_d[j])
