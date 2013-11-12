import csv
import matplotlib.pyplot as plt

file_sel = open("data/features_sel_backward_gap_run_2.csv","wb")
file_writer = csv.writer(file_sel)

file_writer.writerow(["Loop Index","GAP Value","Removed Feature Index",\
        "Removed Feature Name"])
for i in range(len(features_rm)):
    file_writer.writerow([i,score_best[51-i],features_rm[i],\
            features_name_use[features_rm[i]]])

for i in range(2):
    file_writer.writerow([''])

file_writer.writerow(["Last Remaining Feature",features_name_use[bfs[0]]])
file_sel.close()

tmp = []
for i in range(len(features_rm)):
    tmp.append(features_name_use[features_rm[len(features_rm)-1-i]])

plt.figure()
plt.plot(range(1,52),score_best[1:52])
plt.xlabel("The Number of Features")
plt.title("Backward Search with GAP statistic")
plt.show()

