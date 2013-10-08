import csv
import matplotlib.pyplot as plt

file_sel = open("data/features_sel_con_silhouette.csv","wb")
file_writer = csv.writer(file_sel)
file_writer.writerow(["Features removed by redundancy"])
for i in range(len(features_name_con)):
    if i not in features_sel_con:
        file_writer.writerow([features_name_con[i]])
for i in range(3):
    file_writer.writerow([''])
file_writer.writerow(["The following are results of wrapper:"])

file_writer.writerow(["Loop Index","Silhouette Value","Removed Feature Index",\
        "Removed Feature Name"])
for i in range(len(features_rm)):
    file_writer.writerow([i,score_best[51-i],features_rm[i],\
            features_name_con[features_sel_con[features_rm[i]]]])

for i in range(2):
    file_writer.writerow([''])
file_writer.writerow(["Last Remaining Feature",\
        features_name_con[features_sel_con[bfs[0]]]])
file_sel.close()

plt.plot(score_best)
