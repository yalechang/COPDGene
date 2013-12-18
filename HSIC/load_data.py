import pickle
import numpy as np
import csv
from sklearn.preprocessing import scale

# Load dataset from pickle file
file_data = open("/home/changyale/dataset/COPDGene/data_train_continuous.pkl",\
        "rb")
data_con, features_con = pickle.load(file_data)
file_data.close()

# Normalization of the dataset
data = scale(data_con)

# Write the scaled dataset into a csv file
file_csv = open("data_train_continuous.csv","wb")
file_writer = csv.writer(file_csv)
for i in range(data.shape[0]):
    file_writer.writerow(list(data[i,:]))
file_csv.close()

