import csv
import matplotlib.pyplot as plt

file_sel = open("data/features_sel_forward.csv","wb")
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
        "Silhouette(Reference)","Added Feature Index",\
        "Added Feature Name"])
for i in range(len(features_add)-1):
    file_writer.writerow([i,score_best[i],score_best_ref[i],\
            features_add[i],features_name_use[features_add[i]]])
file_sel.close()

tmp = []
for i in range(len(features_add)-1):
    tmp.append(features_name_use[features_add[i]])

plt.figure()
plt.plot(range(1,52),score_best[0:51])
plt.plot(range(1,52),score_best_ref[0:51])
plt.xlabel("The Number of Features")
#plt.xticks(range(1,52),tmp,rotation='vertical',size='small')
plt.title("Forward Search for Continuous Features")
plt.legend(['Silhouette(Original)','Silhouette(Reference)'],loc='upper right')
plt.show()
