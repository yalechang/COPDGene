import numpy as np
import pickle
from sklearn.cluster import spectral_clustering
from draw_similarity_matrix import draw_similarity_matrix
import matplotlib.pyplot as plt
import matplotlib.colorbar as plr
import networkx as nx
import csv
import copy

# Load dataset
file_data = open("data_train.pkl","rb")
info_con, info_dis, gold = pickle.load(file_data)
file_data.close()

# Extract info on continuous dataset
data_con, features_name_con, features_type_con = info_con

# Obtain dataset size
n_instances,n_features = data_con.shape

# Load affinity matrix
# Output of "compute_affinity_hsic_con.py"
# Use Gaussian Kernel with sigma_hsic=1.0
file_hsic = open("mtr_hsic_nhsic.pkl","rb")
mtr_hsic,mtr_nhsic = pickle.load(file_hsic)
file_hsic.close()

# Load features ranking information
file_ranking = open("exp_result/ranking_con_Ref1.csv","rb")
reader = csv.reader(file_ranking)
lines = [line for line in reader]
assert len(lines) == n_features+1

# Extract the list of ordered features from input csv file
features_ordered = {lines[i][3]:i-1 for i in range(1,len(lines))}

# Parameter Settings
##############################################################################
# The number of feature subsets
n_clusters = 10
##############################################################################

# Cluster features using Spectral Clustering
labels = spectral_clustering(mtr_nhsic,n_clusters=n_clusters,n_init=10)
print np.unique(labels)

##############################################################################
## Draw a graph where the nodes are features, the weights of edges represents
## similarity(Normailized HSIC value) between two nodes
#   Step 1: Construct graph according to affinity matrix
# Initialize the graph
g = nx.Graph()

# Create edges
edges = []
for i in range(0,n_features-1):
    for j in range(i+1,n_features):
        edges.append((features_name_con[i],features_name_con[j],mtr_nhsic[i,j]))
g.add_weighted_edges_from(edges)

#   Step 2: Draw the graph constructed as above
# Sepcify the size of the fig, Default(8x8)
fig = plt.figure(figsize=(30,30))

# Specify layout of the graph 
pos = nx.spring_layout(g)

n_colors = n_features
# Create colormap
cm = plt.get_cmap('gist_rainbow')

ax = fig.add_subplot(111)
ax.set_color_cycle([cm(1.*i/n_colors) for i in range(n_colors)])

## Draw nodes
# Colormap
sm = plt.cm.ScalarMappable(cmap=cm,norm=plt.normalize(vmin=0,vmax=1))
sm._A = []

# markers that specify the shapes of nodes
marker = ['o','s','*','8','D','+','h','v','^','<','>','x','p',\
        'H','d','|','_','1','2','3','4'] 
for i in range(n_features):
    # Find the Ranking of this node by comparing with features_ordered
    tmp = features_ordered[features_name_con[i]]
    nx.draw_networkx_nodes(g,pos,
            nodelist=[features_name_con[i]],
            node_size = 300,
            alpha = 1.0,
            node_color = cm(1.*tmp/n_colors),
            node_shape = marker[labels[i]],
            cmap = cm,
            vmin = 0.,
            vmax = 1.,
            labels = features_name_con[i])
    nx.draw_networkx_labels(g,pos,
            font_size = 8,
            font_color = 'k',
            font_weight = 'ultralight',
            alpha = 0.2)
for i in range(n_features-1):
    for j in range(i+1,n_features):
        if mtr_nhsic[i,j] > 0.9:
            nx.draw_networkx_edges(g,pos,
                    edgelist=[(features_name_con[i],features_name_con[j])])
cbar = plt.colorbar(sm)
plt.savefig('exp_result/figures/features_visualization_K'+str(n_clusters)\
        +'.png')

##############################################################################

# Draw similarity matrix
affinity = mtr_nhsic

# Get the number of samples 
n_samples = len(labels)

# number of samples in every cluster
counter = [0]*n_clusters
for i in range(n_samples):
    for j in range(n_clusters):
        if labels[i] == j:
            counter[j] = counter[j]+1

# rank the clusters according to their cluster size
new_counter = copy.copy(counter)
new_cluster_id = range(n_clusters)
for i in range(0,n_clusters-1):
    for j in range(i+1,n_clusters):
        if new_counter[i]<new_counter[j]:
            tmp = new_counter[i]
            new_counter[i] = new_counter[j]
            new_counter[j] = tmp
            tmp1 = new_cluster_id[i]
            new_cluster_id[i] = new_cluster_id[j]
            new_cluster_id[j] = tmp1

# the index of the 1st sample of every cluster
index_start = range(n_clusters)
for i in range(1,n_clusters):
    index_start[i] = sum(new_counter[0:i])

# num is the number of elements in a cluster
# it changes every time, no greater than counter[j] 
num = [0]*n_clusters

# pos_old contain the original location of the new location
pos_old = {}
   
# affinity_order is the new affinity matrix after ordering rows
affinity_order = np.zeros((n_samples,n_samples))
affinity_1 = np.zeros((n_samples,n_samples))
for i in range(n_samples):
    for j in range(len(new_cluster_id)):
        if new_cluster_id[j] == labels[i]:
            tmp = j
            break
    i_new = index_start[tmp]+num[tmp]
    pos_old[i_new] = i
    for j in range(n_samples):
        affinity_order[i_new,j] = affinity[i,j]
    num[tmp] = num[tmp]+1

# affinity_1 is the new affinity matrix after ordering columns 
num = [0]*n_clusters
for j in range(n_samples):
    for i in range(len(new_cluster_id)):
        if new_cluster_id[i] == labels[j]:
            tmp = i
            break
    j_new = index_start[tmp]+num[tmp]
    affinity_1[:,j_new] = affinity_order[:,j]
    num[tmp] = num[tmp]+1

# Output feature clusters
file_features_cluster = open('exp_result/similarity_matrix_K'+str(n_clusters)+'.csv',"wb")
file_writer = csv.writer(file_features_cluster)
for i in range(n_samples):
    file_writer.writerow([i,features_name_con[pos_old[i]]])
file_writer.writerow(new_counter)
print new_counter

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(affinity_1,interpolation='nearest')
#grid(True)
fig.colorbar(im)
plt.savefig('exp_result/figures/similarity_matrix_K'+str(n_clusters)+'.png')

