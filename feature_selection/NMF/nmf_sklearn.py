""" This script will apply NMF on the 4413 x 13 data matrix corresponding to
the 13 selected features from backward search 
"""

print __doc__

import numpy as np
import csv
from sklearn.decomposition import ProjectedGradientNMF
from sklearn.decomposition import NMF
from python.COPDGene.utils.draw_similarity_matrix import draw_similarity_matrix
from sklearn.cluster.spectral import spectral_clustering
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import pairwise_kernels
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans

# Load data matrix
# flag = True, then use SHIFTED dataset;otherwise use original dataset
flag = True
if flag == True:
    file_csv = open("/home/changyale/dataset/COPDGene/data_sel_backward_"+\
            "gap4_SHIFTED.csv","rb")
    reader = csv.reader(file_csv)
    lines = [line for line in reader]
    file_csv.close()

    data = np.zeros((len(lines)-1,len(lines[0])-1))
    for i in range(data.shape[0]):
        data[i,:] = np.array(lines[i+1][1:len(lines[i+1])])
    print "Use SHIFTED dataset"
else:
    file_csv = open("/home/changyale/dataset/COPDGene/data_sel_backward_"+\
            "gap4.csv","rb")
    reader = csv.reader(file_csv)
    lines = [line for line in reader]
    file_csv.close()
    
    data = np.zeros((len(lines)-2,len(lines[0])-1))
    for i in range(data.shape[1]):
        data[i,:] = np.array(lines[i+2][1:len(lines[i+1])])
    print "Use ORIGINAL dataset"

# Apply NMF on the dataset
# n_components: Number of components, if n_components is not set all are kept
# init: 'nndsvd'(better for sparseness)Nonnegative Double SVD
#       'nndsvda'(better when sparsity is not desired)NNDSVD with zeros filled
#       with the average of data matrix X
#       'nndsvdar'(generally faster,less accurate alternative to NNDSVDa for
#       when sparisity is not desried)
#       'random':non-negative random matrices
# sparseness: 'data'|'components'|None,default:None(where to enforce sparsity
#               in the model)
# beta: degree of sparseness, if sparseness is not None, Larger values mean
#       more sparseness. Default value is 1.
# eta: degree of correctness to maintain, if sparsity is not None. Smaller
#       values mean larger error. Default value is 0.1
# tol: tolerance value used in stopping conditions, default value is 1e-4
# max_iter: number of iterations to compute, default value is 200
# nls_max_iter: number of iterations in NLS subproblem, default value is 2000
# random_state: init or RandomState, random number generator seed control
# Define the parameters for the model

# Number of latent factors for features
n_factors = 6
model = NMF(n_components=n_factors,init='nndsvda',sparseness=None,\
        beta=1,eta=0.1,tol=0.0001,max_iter=200,nls_max_iter=2000,\
        random_state=None)
# Fit model to the data and return n_samples x n_components matrix
data_factors = model.fit_transform(data)
print(["Reconstructino Error: ",model.reconstruction_err_])
print("Model Components:")
mtr_tmp = model.components_.T
for i in range(mtr_tmp.shape[0]):
    print lines[0][1+i],
    for j in range(mtr_tmp.shape[1]-1):
        print ' & ',"%.2f" % mtr_tmp[i,j],
    print ' & ',"%.2f" % mtr_tmp[i,mtr_tmp.shape[1]-1],'\\\\'


sigma_rbf = 2.0
# Number of clusters for samples
K = n_factors
# Normalization of transformed data(matrix H)
data_factors = scale(data_factors)

# Choose clustering method:'kmeans' or 'spectral'
method = 'kmeans'
affinity = pairwise_kernels(data_factors,data_factors,metric='rbf',\
        gamma=1./(2*sigma_rbf**2))

if method == 'spectral':
    clf = SpectralClustering(n_clusters=K,affinity='precomputed')
    clf.fit(affinity)
    labels_predict = clf.labels_
else:
    clf = KMeans(n_clusters=K,init='random',n_jobs=-1)
    clf.fit(data_factors)
    labels_predict = clf.labels_

# Show similarity matrix 
#draw_similarity_matrix(affinity,labels_predict,K)

# Show basis vector matrix
#mtr_tmp = mtr_tmp/np.sum(mtr_tmp,axis=1)[:,None]
plt.pcolor(mtr_tmp)

plt.xticks(range(1,1+n_factors))
plt.ylim(13,0)
plt.yticks(list(np.arange(13,0,-1)))
plt.colorbar()
plt.xlabel("Latent Factors")
plt.ylabel("Features")
plt.title("Latent Factor Matrix")
plt.show()
