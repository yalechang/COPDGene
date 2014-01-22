""" Compute affinity matrix between continuous features. The similarity is
computed through Normalized HSIC of pairwise features.

INPUT
-----
data_train_continuous.pkl: Size (4413 X 63)

OUTPUT
------
mtr_hsic_nhsic_con.pkl: Similarity matrix between features.
"""

print __doc__

import pickle
import numpy as np
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import rbf_kernel
from time import time
import csv

# Load dataset
file_data = open("/home/changyale/dataset/COPDGene/data_train_continuous.pkl",\
        "rb")
data_con,features_con = pickle.load(file_data)
file_data.close()

# Parameter setting
##############################################################################
sigma_hsic = 1.0
##############################################################################

# Normalization of the dataset
data = scale(data_con)

# Obtain dataset size
n_instances,n_features = data.shape

# Compute HSIC matrix
mtr_hsic = np.zeros((n_features,n_features))
arr_h = np.eye(n_instances)-1./n_instances
print(["Row","Col","RunningTime"])
for i in range(n_features):
    tmp = data[:,i].reshape(n_instances,1)
    arr_l = rbf_kernel(tmp,tmp,1./(2*sigma_hsic**2))
    arr_hlh = np.dot(np.dot(arr_h,arr_l),arr_h)
    for j in range(i,n_features):
        t0 = time()
        tp = data[:,j].reshape(n_instances,1)
        arr_k = rbf_kernel(tp,tp,1./(2*sigma_hsic**2))
        mtr_hsic[j,i] = 1./(n_instances-1)**2*np.trace(np.dot(arr_k,arr_hlh))
        mtr_hsic[i,j] = mtr_hsic[j,i]
        t1 = time()
        print([i,j,(t1-t0)/60])

# Compute Normalized HSIC matrix
mtr_nhsic = np.zeros((n_features,n_features))
for i in range(n_features):
    for j in range(i,n_features):
        mtr_nhsic[i,j] = mtr_hsic[i,j]/np.sqrt(mtr_hsic[i,i]*mtr_hsic[j,j])
        mtr_nhsic[j,i] = mtr_nhsic[i,j]

file_dump = open("data/mtr_hsic_nhsic_con.pkl","wb")
pickle.dump([mtr_hsic,mtr_nhsic],file_dump)
file_dump.close()
