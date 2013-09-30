import pickle
import numpy as np
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import rbf_kernel
from time import time
import csv
import copy

# Load dataset
file_data = open("data_train.pkl","rb")
info_con,info_dis,gold = pickle.load(file_data)
file_data.close()

# Extract info on continuous dataset
data_con, features_name_con, features_type_con = info_con

# Parameter setting
# Part 1: Choose subsets from dataset according to Reference 1,2,3
#       Reference 1: Use Gold 0,1,2,3,4. Discard U type
#       Reference 2: Use Gold 0 as COPD 0 and Gold 2,3,4 as COPD 1
#       Reference 3: Use Gold 0 as SevereCOPD 0 and Gold 3,4 as SevereCOPD 1
# Part 2: Try different value for scale parameter in HSIC kernel matrix
# $ k(\mathbf{X}_i,\mathbf{X}_j) = \exp(-\frac{\vectornorm{\mathbf{X}_i..
# -\mathbf{X}_j}^2}{2\sigma^2}) $
##############################################################################
flag_reference = 3
sigma_hsic = [0.5,1.0,2.0,5.0,10.0]
##############################################################################
data = []
gold_left = []
if flag_reference == 1:
    for i in range(data_con.shape[0]):
        if gold[i] in [0,1,2,3,4]:
            data.append(data_con[i,:])
            gold_left.append(float(gold[i]))
if flag_reference == 2:
    for i in range(data_con.shape[0]):
        if gold[i] in [0,2,3,4]:
            data.append(data_con[i,:])
            if gold[i] == 0:
                gold_left.append(0.)
            else:
                gold_left.append(1.)
if flag_reference == 3:
    for i in range(data_con.shape[0]):
        if gold[i] in [0,3,4]:
            data.append(data_con[i,:])
            if gold[i] == 0:
                gold_left.append(0.)
            else:
                gold_left.append(1.)

# Normalization of the dataset
data = scale(np.array(data))
gold_left = scale(np.array(gold_left))
print data.shape

# Get the size of the dataset
n_instances,n_features = data.shape

# Every column is the result of a sigma_hsic value
hsic_val = np.zeros((n_features,len(sigma_hsic)))

# Every column is the Normalized HSIC values of a sigma_hsic
nhsic_val = np.zeros((n_features,len(sigma_hsic)))

# Compute hsic for value for each feature
for i in range(len(sigma_hsic)):
    print ['sigma_hsic:',sigma_hsic[i]]
    print ['Index','FeatureName','NHSIC_Val','RunningTime']
    # Compute the kernel matrix for GOLD
    tmp = gold_left.reshape(n_instances,1)
    arr_l = rbf_kernel(tmp,tmp,1./(2*sigma_hsic[i]**2))
    arr_h = np.eye(n_instances)-1./n_instances
    arr_hlh = np.dot(np.dot(arr_h,arr_l),arr_h)
    # HSIC(Y,Y)
    tmp_hsic_i = 1./(n_instances-1)**2*np.trace(np.dot(arr_l,arr_hlh))
    for j in range(n_features):
        t0 = time()
        tmp = data[:,j].reshape(n_instances,1)
        arr_k = rbf_kernel(tmp,tmp,1./(2*sigma_hsic[i]**2))
        hsic_val[j,i] = 1./(n_instances-1)**2*np.trace(np.dot(arr_k,arr_hlh))
        # HSIC(X[j],X[j])
        tmp_hsic_j = 1./(n_instances-1)**2*np.trace(np.dot(arr_k,np.dot(arr_h,\
                np.dot(arr_k,arr_h))))
        t1 = time()
        # NormalizedHSIC(X[j],Y)
        nhsic_val[j,i] = hsic_val[j,i]/np.sqrt(tmp_hsic_i*tmp_hsic_j)
        print [j,features_name_con[j],nhsic_val[j,i],(t1-t0)/60]

# Sort the NHSIC value
# Sort the NHSIC value
order_nhsic = np.array(range(n_features)*len(sigma_hsic)).\
        reshape(len(sigma_hsic),n_features).T

for i in range(len(sigma_hsic)):
    for j in range(n_features-1):
        for k in range(j+1,n_features):
            if nhsic_val[j,i]<nhsic_val[k,i]:
                tp1 = nhsic_val[j,i]
                nhsic_val[j,i] = nhsic_val[k,i]
                nhsic_val[k,i] = tp1
                tp2 = order_nhsic[j,i]
                order_nhsic[j,i] = order_nhsic[k,i]
                order_nhsic[k,i] = tp2
# Write ranking result into a csv file
file_ranking_con = open('ranking_con_Ref'+str(flag_reference)+'.csv',"wb")
file_writer = csv.writer(file_ranking_con)
tp = []
for i in range(len(sigma_hsic)):
    tp.append([sigma_hsic[i],"FeaturesName"])
    tp.append([sigma_hsic[i],"HSICValue"])
file_writer.writerow(['FeaturesRanking\SigmaValue']+tp)
for j in range(n_features):
    tp = []
    for i in range(len(sigma_hsic)):
        tp.append(features_name_con[order_nhsic[j,i]])
        tp.append(nhsic_val[j,i])
    file_writer.writerow([j]+tp)
file_ranking_con.close()

