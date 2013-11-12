import csv
import pickle
import numpy as np

# Read affinity matrix of discrete features from csv file
csvfile = open("mtr_hsic_dis.csv","rb")
reader = csv.reader(csvfile)
lines = [line for line in reader]
n_row = len(lines)
n_col = len(lines[0])

# array
mtr_hsic_dis = np.double(np.array(lines))
csvfile.close()

# process normalized HSIC
csvfile = open("mtr_nhsic_dis.csv","rb")
reader = csv.reader(csvfile)
lines = [line for line in reader]
mtr_nhsic_dis = np.double(np.array(lines))
csvfile.close()

file_result = open("mtr_hsic_nhsic_dis.pkl","wb")
pickle.dump([mtr_hsic_dis,mtr_nhsic_dis],file_result)
file_result.close()

