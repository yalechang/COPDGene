""" This script will make plots for data generated from gap4, including adding
labels for Y axis and draw circle or line to show the maximal.
"""

print __doc__

import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt

# Load training set
file_data_train = open("/home/changyale/dataset/COPDGene/data_"+\
        "train_continuous.pkl","rb")
data_con_use,id_use,features_name_use = pickle.load(file_data_train)

# final selected feature set after backward search
features_sel_index = [55,54,26,27,32,53,19,17,33,25,18,28,50]

# data and features corresponding to the selected feature set by backward
# search
data_sel = data_con_use[:,features_sel_index]

features_sel_name = []
for i in features_sel_index:
    features_sel_name.append(features_name_use[i])

# Write data_sel into a csv file
file_sel = open("gap_4/data_sel_backward_gap4.csv","wb")
file_writer = csv.writer(file_sel)

file_writer.writerow(["Features Index"]+features_sel_index)
file_writer.writerow(["Features Name"]+features_sel_name)
for i in range(data_sel.shape[0]):
    file_writer.writerow([id_use[i]]+list(data_sel[i]))
file_sel.close()

