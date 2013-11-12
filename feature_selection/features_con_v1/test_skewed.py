import pickle
import numpy as np
from time import time
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from python.COPDGene.utils.sample_wr import sample_wr

############################# Parameter Setting ##############################
# The number of clusters in KMeans
K = 4
##############################################################################

t0 = time()
# Load training set
file_data_train = open("/home/changyale/dataset/COPDGene/data_train.pkl","rb")
info_con,info_dis,gold = pickle.load(file_data_train)
file_data_train.close()
data_con, features_name_con, features_type_con = info_con

# Choose only 'continuous' features for backward search
data_con_use = []
features_name_use = []
for j in range(len(features_type_con)):
    if features_type_con[j] == 'continuous':
        data_con_use.append(data_con[:,j])
        features_name_use.append(features_name_con[j])
data_con_use = np.array(data_con_use).T

# Forward search for continuous features
# Normalization of the continuous dataset
data = scale(data_con_use)
n_instances, n_features = data.shape

fs = [24,17,16,21,8,9,22,15,18,37]

for i in range(len(fs)):
    fs_use = fs[0:i+1]
    data_use = data[:,fs_use]
    #estimator = KMeans(init='random',n_clusters=K,n_init=10,n_jobs=-1)
    estimator = SpectralClustering(n_clusters=K,gamma=1.0,n_init=10)
    estimator.fit(data_use)
    data_use_labels = estimator.labels_
    score_sil = silhouette_score(data_use,data_use_labels,metric='euclidean')

    freq = [0]*K
    for i in range(len(data_use_labels)):
        freq[data_use_labels[i]] += 1
    print i+1,freq,score_sil
