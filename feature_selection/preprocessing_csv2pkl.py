""" This script will read data stored in a csv format file and write into a
pickle file.

Parameters
----------
dataset_complete_knn_dropoutlier_11-4-13.csv: the imputated dataset after
dropping out outliers. We need to read data from this csv file and store the
data into a pickle file.

Returns
-------
dataset_complete_knn_dropoutlier_11-4-13.pkl
"""
print __doc__

import numpy as np
import csv
import pickle

# Read csv file
file_csv = open("/home/changyale/dataset/COPDGene/dataset_complete_knn_windsor_11-18-13.csv","rb")
reader = csv.reader(file_csv)
lines = [line for line in reader]
mtr = np.array(lines)
file_csv.close()

dataset = mtr[1:mtr.shape[0],1:mtr.shape[1]]
features_name = mtr[0,1:mtr.shape[1]]
features_type = ['Unspecified']*mtr.shape[1]

file_pkl = open("/home/changyale/dataset/COPDGene/dataset_complete_knn_windsor_11-18-13.pkl","wb")
pickle.dump([dataset,features_name,features_type],file_pkl)
file_pkl.close()

