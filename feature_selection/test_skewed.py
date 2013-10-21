import pickle
import numpy as np
from time import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from python.COPDGene.utils.sample_wr import sample_wr

K = 4

data_use = data[:,[1]]
estimator = KMeans(init='random',n_clusters=K,n_init=10,n_jobs=-1)
estimator.fit(data_use)
data_use_labels = estimator.labels_
score_sil = silhouette_score(data_use,data_use_labels,metric='euclidean')

freq = [0]*K
for i in range(len(data_use_labels)):
    freq[data_use_labels[i]] += 1

