import csv
import matplotlib.pyplot as plt

file_sel = open("data/features_sel_forward_gap_run_1.csv","wb")
file_writer = csv.writer(file_sel)

file_writer.writerow(["Loop Index","GAP Value","Added Feature Index",\
        "Added Feature Name"])
for i in range(len(features_add)):
    file_writer.writerow([i,score_best[i],features_add[i],\
            features_name_use[features_add[i]]])
file_sel.close()

tmp = []
for i in range(len(features_add)):
    tmp.append(features_name_use[features_add[i]])

plt.figure()
plt.plot(range(1,53),score_best)
plt.xlabel("The Number of Features")
plt.title("Forward Search With GAP Statistic and Supervision")
plt.show()
