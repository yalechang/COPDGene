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

########################### Parameter Settings################################
# Number of clusters of samples for applying spectral clustering
n_clusters = 4
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

"""
# We still need to remove "RandomGroupCode"
del features_name_use[index_RandomGroupCode]
del features_type_use[index_RandomGroupCode]


# Random sample with replacement from data_train to form a reference dataset
data_train_ref = np.zeros((data_train.shape[0],data_train.shape[1]))
for j in range(data_train_ref.shape[1]):
    tp_index = sample_wr(range(data_train_ref.shape[0]),data_train_ref.shape[0])
    for i in range(len(tp_index)):
        data_train_ref[i,j] = data_train[tp_index[i],j]
# Label data_train as class 0 and data_train_ref as class 1
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

# Random Forest Classifier
t0 = time()
clf = RandomForestClassifier(n_estimators=1000,criterion='gini',max_depth=None,\
        min_samples_split=2,min_samples_leaf=1,max_features='auto',\
        bootstrap=True,oob_score=True,n_jobs=-1,random_state=None,\
        verbose=0,compute_importances=None)
clf.fit(data_use,labels)
t1 = time()
print(["Random Forest Running Time(min):",(t1-t0)/60])

# Compute similarity matrix for the training set
leaf_index = clf.apply(data_train)
t0 = time()
mtr_affinity = compute_similarity_from_leaf_index(np.double(leaf_index))
t1 = time()
print(["Similarity matrix computation time(min)",(t1-t0)/60])

# Store the similarity matrix for future use
#file_result = open("mtr_affinity_random_forest.pkl","wb")
#pickle.dump(mtr_affinity,file_result)
#file_result.close()

# Apply spectral clustering given the similarity matrix as input. Note here
# since we already have similarity matrix, we don't have to specify the scale
# parameter to construct the kernel matrix
t0 = time()
labels_train = spectral_clustering(mtr_affinity,n_clusters=n_clusters,\
        n_components=None,eigen_solver=None,random_state=None,n_init=10,\
        k=None,eigen_tol=0.0,assign_labels='kmeans',mode=None)
t1 = time()
print(["Spectral Clustering time(min)",(t1-t0)/60])

# Get the eigenvalues associated with the similarity matrix

features_importance = clf.feature_importances_
tp = copy.copy(features_name)

# Rank features according to features_importance in descending order
for i in range(len(tp)-1):
    for j in range(i+1,len(tp)):
        if features_importance[i] < features_importance[j]:
            temp = features_importance[i]
            features_importance[i] = features_importance[j]
            features_importance[j] = temp
            temp = tp[i]
            tp[i] = tp[j]
            tp[j] = temp

# Write the ranked features into a csv file
file_result = open("data/features_importance.csv","wb")
file_writer = csv.writer(file_result)
file_writer.writerow(["Rank","Feature Name","Feature Score"])
for i in range(len(tp)):
    file_writer.writerow([i+1,tp[i],features_importance[i]])
file_result.close()

# Construct Laplacian from similarity matrix

# Degree matrix
mtr_d = sum(mtr_affinity)

# Laplacian matrix
n_instances = mtr_affinity.shape[0]
mtr_l = np.zeros((n_instances,n_instances))
for i in range(n_instances):
    for j in range(n_instances):
        mtr_l[i,j] = 1./np.sqrt(mtr_d[i])*mtr_affinity[i,j]*1./np.sqrt(mtr_d[j])
"""
