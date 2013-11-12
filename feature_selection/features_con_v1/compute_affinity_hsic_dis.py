""" Compute affinity matrix between discrete features. The similarity is
computed through Normalized HSIC of pairwise features.
"""

import pickle
import numpy as np
from time import time
import csv

# Load dataset
file_data = open("/home/changyale/dataset/COPDGene/data_train.pkl","rb")
info_con,info_dis,gold = pickle.load(file_data)
file_data.close()

# Extract information of discrete dataset
data, features_name_dis, features_type_dis = info_dis

# Obtain dataset size
n_instances,n_features = data.shape

# Compute HSIC matrix
mtr_hsic = np.zeros((n_features,n_features))
arr_h = np.eye(n_instances)-1./n_instances
print(["Row","Col","RunningTime"])
for i in range(n_features):
    t0 = time()
    print("BegingTest")
    tmp = np.array([data[:,i]]*n_instances)
    print tmp.shape
    print(time()-t0)
    arr_l = np.double(tmp==tmp.T)
    print arr_l.shape
    print(time()-t0)
    arr_hlh = np.dot(np.dot(arr_h,arr_l),arr_h)
    print arr_hlh.shape
    print(time()-t0)
    for j in range(i,n_features):
        t0 = time()
        tp = np.array([data[:,j]]*n_instances)
        arr_k = np.double(tp==tp.T)
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

file_dump = open("data/mtr_hsic_nhsic_dis.pkl","wb")
pickle.dump([mtr_hsic,mtr_nhsic],file_dump)
file_dump.close()
