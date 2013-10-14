import numpy as np
from sklearn.preprocessing import scale
#import matplotlib.pyplot as plt
import pickle
from time import time
#from scipy.io import savemat

from read_copdgene_data import read_copdgene_data
from parse_copdgene_data import parse_copdgene_data
from hsic import hsic

t0 = time()

# Number of views, here we mainly consider feature sets F1,F2,where F1
# contains "Emph", F2 contains "WallArea"
M = 2

# scale parameter for kernel in computing HSIC
sigma_hsic = 1.0

# Read original data from the file, there may be missing data
filename = "Training.csv"
[data_raw,features,case_ids] = read_copdgene_data(filename)

# Select the data of interest
features_of_interest = features[1:17]
cases_of_interest = case_ids
parse_result = parse_copdgene_data(data_raw,features,case_ids,
                                  features_of_interest,cases_of_interest)

# data_of_interest contains the case_ids which have no missing data of 
# corresponding features of interest
data_of_interest = parse_result[0]
found_features_of_interest = parse_result[2]
print features_of_interest

"""
data_save = {}
data_save['data_of_interest'] = data_of_interest
savemat('Training.mat',data_save)
"""

# find "Gold" from data_of_interest, this is done by consider the values of 
# two features:FEV1pp_utah(80,50,30), FEV1_FVC_utah(0.7)
# ============================================================================
for i in range(len(features_of_interest)):
    if features_of_interest[i] == 'FEV1pp_utah':
        index_1 = i
    if features_of_interest[i] == 'FEV1_FVC_utah':
        index_2 = i
print index_1,index_2

temp_1 = data_of_interest.shape[0]
temp_2 = 0
index_gold = range(temp_1)
for i in range(temp_1):
    if data_of_interest[i,index_2]>=0.7:
        if data_of_interest[i,index_1]>=80:
            index_gold[i] = 0
        else:
            index_gold[i] = 'U'
    if data_of_interest[i,index_2]<0.7:
        temp_2 = temp_2+1
        if data_of_interest[i,index_1]>=80:
            index_gold[i] =1
        if data_of_interest[i,index_1]<80 and data_of_interest[i,index_1]>=50:
            index_gold[i] = 2
        if data_of_interest[i,index_1]<50 and data_of_interest[i,index_1]>=30:
            index_gold[i] = 3
        if data_of_interest[i,index_1]<30:
            index_gold[i] = 4

data_1 = np.zeros((temp_2,len(features_of_interest)))
gold = range(temp_2)
ii = 0
for i in range(temp_1):
    if not(index_gold[i]==0 or index_gold[i]=='U'):
        for j in range(len(features_of_interest)):
            data_1[ii,j] = data_of_interest[i,j]
        gold[ii] = index_gold[i]
        ii = ii+1

n_samples,n_features = data_1.shape
print("n_samples: %d, \t n_features: %d"
     % (n_samples,n_features))

#print gold

# normalization of original data
data = scale(data_1)

output = open("data_gold_training.pkl","wb")
pickle.dump([data,gold],output)
output.close()

x_1 = np.ones((n_samples,2))
x_2 = np.ones((n_samples,2))

# calculate the similarity between a pair of features fi and fj,using HSIC
# matrix_hsic[i,j] = HSIC(fi,fj) 
matrix_hsic = np.zeros((n_features,n_features))
for i in range(n_features):
    print [i,time()-t0]
    for j in range(i+1,n_features):
        x_1[:,0] = data[:,i]
        x_2[:,0] = data[:,j]
        matrix_hsic[i,j] = hsic(x_1,x_2,sigma_hsic)
        matrix_hsic[j,i] = matrix_hsic[i,j]
        #print [i,j,time()-t0]

print time()-t0

output = open("matrix_hsic_training.pkl","wb")
pickle.dump(matrix_hsic,output)
output.close()


