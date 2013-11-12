import csv
import matplotlib.pyplot as plt

file_sel = open("data/features_sel_continuous_silhouette.csv","wb")
file_writer = csv.writer(file_sel)

"""
file_writer.writerow(["Features removed by redundancy"])
for i in range(len(features_name_con)):
    if i not in features_sel_con:
        file_writer.writerow([features_name_con[i]])
for i in range(3):
    file_writer.writerow([''])
file_writer.writerow(["The following are results of wrapper:"])
"""

file_writer.writerow(["Loop Index","Silhouette(Original)",\
        "Silhouette(Reference)","Removed Feature Index",\
        "Removed Feature Name"])
for i in range(len(features_rm)):
    file_writer.writerow([i,score_best[51-i],score_best_ref[51-i],\
            features_rm[i],features_name_use[features_rm[i]]])

for i in range(2):
    file_writer.writerow([''])
file_writer.writerow(["Last Remaining Feature",features_name_use[bfs[0]]])
file_sel.close()

tmp = []
for i in range(len(features_rm)):
    tmp.append(features_name_use[features_rm[len(features_rm)-1-i]])

plt.figure()
plt.plot(range(1,52),score_best[1:52])
plt.plot(range(1,52),score_best_ref[1:52])
plt.xlabel("The Number of Features")
plt.xticks(range(1,52),tmp,rotation='vertical',size='small')
plt.title("Backward Search for Continuous Features")
plt.legend(['Silhouette(Original)','Silhouette(Reference)'],loc='upper right')
plt.show()
