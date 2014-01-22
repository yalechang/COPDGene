""" Draw plots from data in gap_4/features_sel_backward_gap_run_1.csv
Label Y axis
"""

import csv
import matplotlib.pyplot as plt
import numpy as np

# Load GAP values
file_csv = open("gap_4/features_sel_backward_gap_run_1.csv","rb")
reader = csv.reader(file_csv)
lines = [line for line in reader]
file_csv.close()

gap_value = []
fs_index = [55]
fs_name = ['pre_FVC']

for i in range(61):
    gap_value.append(lines[62-i][1])
    fs_index.append(lines[62-i][2])
    fs_name.append(lines[62-i][3])
gap_value.append(lines[1][1])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(1,63),gap_value)
ax.plot((13,13),(-0.05,gap_value[12]),'r--')
ax.annotate('MAX(13, 0.2658)',xy=(13,gap_value[12]))
plt.xlabel("The Number of Features")
plt.ylabel("GAP statistic(Clustering Quality)")
plt.title("Backward Search with GAP statistic")
plt.show()

