"""
This script implement imputation of 10,000 dataset based on mean imputation
and KNN-based imputation after the process of imputation according to the 
questionnaires and domain knowledge provided by experts.

Inputs:
-------
data_imputation_questionnaire.pkl
    pickle file containing the dataset after imputation according to the
    questionnaires and experts' domain knowledge. There are still some missing
    values in this dataset.

Outputs:
dataset_complete_mean.pkl
    pickle file containing the dataset after imputation according to the
    specified algorithm.
dataset_complete_mean.csv
    csv file containing the same information with the python pickle file
"""
print __doc__

import numpy as np
from python.COPDGene.utils.heom import heom
from python.COPDGene.utils.heom import heom_array
from time import time
from python.COPDGene.utils.sorting_top import sorting_top
import pickle
import csv
import copy

def imputation_algorithm(data,features_type,algorithm='mean',**kwargs):
    """ This function implement various imputation algorithms

    Parameters
    ----------
    data: array, shape(n_instances,n_features)
         array containing the dataset, there are some missing values in this
         array and we might assume the missingness are at random
         also note that this dataset contains mixed type data, so the dataset
         is stored as an array of lists

    features_type: list, len(n_features)
        list containing the type for each feature 

    algorithm: str
        indicator of the algorithm used for imputation. The possible value
        could be any element in {'mean','hot-deck','som','knn'}

    **kwargs: keyword arguments
        parameters for algorithms specified by parameter "algorithm"

    Returns
    -------
    dataset: array, shape(n_instances,n)
        dataset after imputation according to specified imputation algorithm
    """
    n_instances,n_features = data.shape
    dataset = np.empty((n_instances,n_features),dtype=list)
    
    # indicate if there're missing values for this feature
    index_missing = [0]*n_features
    for j in range(n_features):
        for i in range(n_instances):
            if data[i,j] == '':
                index_missing[j] = 1
            else:
                # copy existing values of data to the output array
                dataset[i,j] = data[i,j]

    # IMPUTATION using mean
    if algorithm == 'mean':
        for j in range(n_features):
            # there's missingness for j-th feature
            if index_missing[j] == 1:
                # use mean for real-valued features
                if features_type[j] in ['continuous','interval']:
                    temp_sum = 0
                    temp_num = 0
                    for i in range(n_instances):
                        if data[i,j] not in ['',None]:
                            temp_sum += float(data[i,j])
                            temp_num += 1
                    temp_mean = temp_sum*1./temp_num
                    for i in range(n_instances):
                        if data[i,j] in ['',None]:
                            dataset[i,j] = temp_mean

                # use mode for categorical-valued(including binary) features
                elif features_type[j] in ['categorical','binary','ordinal']:
                    temp_type = []
                    temp_freq = []
                    for i in range(n_instances):
                        if data[i,j] not in temp_type:
                            temp_type.append(data[i,j])
                            temp_freq.append(1)
                        else:
                            for k in range(len(temp_type)):
                                if temp_type[k] == data[i,j]:
                                    temp_freq[k] += 1
                    assert len(temp_type) == len(temp_freq)
                    temp_freq_max = max(temp_freq)
                    for k in range(len(temp_freq)):
                        if temp_freq[k] == temp_freq_max:
                            temp_mode = temp_type[k]
                    # imputation using mode
                    for i in range(n_instances):
                        if data[i,j] in ['',None]:
                            dataset[i,j] = temp_mode
    # IMPUTATION using KNN
    elif algorithm == 'knn':
        t0 = time()

        # Load the array containing pairwise HEOM distance
        # To save space we could store the array using a list of lists
        csvfile = open("mtr_heom.csv","rb")
        reader = csv.reader(csvfile)
        mtr_heom = [line for line in reader]
        csvfile.close()

        for i in range(n_instances):
            # Location of missing values
            loc_missing = []
            # Find location of missing values for i-th sample
            for j in range(n_features):
                if data[i,j] in ['',None]:
                    loc_missing.append(j)
            # Compute HEOM distance between samples with missing values and its
            # KNN that don't have missing values at attributes to be imputed
            if len(loc_missing)>0:
                print(["Sample Number:"+str(i),"Time:"+str((time()-t0)/60)])
                print(["Number of Missing Values:",len(loc_missing)])
                # HEOM distance between i-th sample and other samples
                dist_heom = []
                # row number of samples corresponding to values in dist_heom
                row_dist_heom = []
                for k in range(n_instances):
                    # flag indicating if there're missing values in k-th 
                    # sample at attributes to be imputed
                    flag = False
                    for j in range(len(loc_missing)):
                        if data[k,loc_missing[j]] in ['',None]:
                            flag = True
                            break
                        
                    # There're no missing values in k-th sample at attributes
                    # to be imputed of i-th sample
                    if flag == False:
                        dist_heom.append(mtr_heom[i][k])
                        row_dist_heom.append(k)

                # Rank values in dist_heom, find the first K smallest values
                assert len(kwargs.keys()) == 1
                # The number of nearest neighbors
                k_knn = kwargs.keys()[0]
                # If the number of neighbors is less than k_knn,then imputation
                # using mean/mode of the whole column
                knn = min(k_knn,len(dist_heom))
                if k_knn>len(dist_heom):
                    print i,len(dist_heom),"#neighbors<k_knn"
                else:
                    print i,len(dist_heom)
                    dist_heom,row_dist_heom = sorting_top(dist_heom,knn)
                    # Impute missing values using neighbors
                    for j in range(len(loc_missing)):
                        if features_type[loc_missing[j]] in ['binary','categorical','ordinal']:
                            weights = range(knn)
                            for k in range(knn):
                                if dist_heom[knn-1] != dist_heom[0]:
                                    weights[k] = (dist_heom[knn-1]-dist_heom[k])/\
                                            (dist_heom[knn-1]-dist_heom[0])
                                else:
                                    weights[k] = 1
                            max_weights = max(weights)
                            for k in range(knn):
                                if max_weights == weights[k]:
                                    dataset[i,loc_missing[j]] = \
                                            data[row_dist_heom[k],loc_missing[j]]

                        if features_type[loc_missing[j]] in ['interval','continuous']:
                            weights = range(knn)
                            temp_sum = 0
                            for k in range(knn):
                                weights[k] = 1./(dist_heom[k]**2)
                                temp_sum += weights[k]*float(data[row_dist_heom[k],\
                                        loc_missing[j]])
                            dataset[i,loc_missing[j]] = temp_sum/knn

    # if the input algorithm is not based on mean/mode 
    else:
        print "NOT DONE YET, please choose 'mean' or 'knn' "

    return dataset

if __name__ == "__main__":
    import pickle
    import csv

    data_features,features_type = \
            pickle.load(open("data_imputation_questionnaire.pkl","rb"))
    n_instances,n_features = data_features.shape
    features_name = data_features[0,:]
    data_whole = data_features[1:n_instances,:]
    n_instances,n_features = data_whole.shape
    
    tp1,tp2,tp3,tp4,features_include = \
            pickle.load(open("info_missing_type_include.pickle","rb"))
    for i in range(len(tp1)):
        assert tp1[i] == features_name[i]
   
    # Extract dataset by only choosing included features
    features_name_include = []
    features_type_include = []
    data = np.empty((n_instances,sum(features_include)),dtype=list)
    index = 0
    for j in range(len(features_include)):
        if features_include[j] == True:
            for i in range(n_instances):
                data[i,index] = data_whole[i,j]
            index += 1
            features_name_include.append(features_name[j])
            features_type_include.append(features_type[j])
    
    tp_features_name_include = copy.copy(features_name_include)

    # Compute the percentage of missingness of included features
    n_instances,n_features = data.shape
    n_missing = 0
    for i in range(n_instances):
        for j in range(n_features):
            if data[i,j] == '':
                n_missing += 1
    percent = n_missing*1./(n_instances*n_features)
    #print n_missing,percent

    # Save a pickle file of dataset containing only included features
    #pickle.dump([data,features_name_include,features_type_include],\
    #        open("data_include.pkl","wb"))
    
    # Specify the method to use
    #==================================================================
    method = 'mean'
    #==================================================================

    # Imputation use chosen imputation algorithm
    tp_data = imputation_algorithm(data,features_type_include,\
            algorithm=method)

    # Merge 'OthFind_Bronchiectasis' and 'HighConcerns_Bronchiec' since they
    # don't overlap entirely. These two features are both binary features, we
    # could merge them using OR operation.
    for j in range(n_features):
        if features_name_include[j] == 'OthFind_Bronchiectasis':
            index_OthFind_Bronchiectasis = j
        if features_name_include[j] == 'HighConcerns_Bronchiec':
            index_HighConcerns_Bronchiec = j
    #print(['OthFind_Bronchiectasis',index_OthFind_Bronchiectasis])
    #print(['HighConcerns_Bronchiec',index_HighConcerns_Bronchiec])
    
    dataset = np.empty((n_instances,n_features-1),dtype=list)
    index_dataset = 0
    for j in range(n_features):
        if j !=index_HighConcerns_Bronchiec and \
                j!=index_OthFind_Bronchiectasis:
            dataset[:,index_dataset] = tp_data[:,j]
            index_dataset += 1
    # OR operation 
    for i in range(n_instances):
        if data[i,index_OthFind_Bronchiectasis]=='1' or \
                data[i,index_HighConcerns_Bronchiec ]=='1':
            dataset[i,index_dataset] = '1'
        else:
            dataset[i,index_dataset] = '0'

    # Change features information accordingly
    print(features_name_include[index_OthFind_Bronchiectasis])
    print(features_name_include[index_HighConcerns_Bronchiec])
    assert index_OthFind_Bronchiectasis < index_HighConcerns_Bronchiec
    del features_name_include[index_OthFind_Bronchiectasis]
    del features_name_include[index_HighConcerns_Bronchiec-1]
    del features_type_include[index_OthFind_Bronchiectasis]
    del features_type_include[index_HighConcerns_Bronchiec-1]
    features_name_include.append('OthFind_HighConcerns_Bronchiec')
    features_type_include.append('binary')

    print dataset.shape 
    # Save a copy in python pickle format 
    pickle.dump([dataset,features_name_include,features_type_include],\
            open("dataset_complete_"+method+".pkl","wb"))
    
    # Save a copy in csv format
    file_name_result = "dataset_complete_"+method+".csv"
    result = open(file_name_result,'wb')
    file_writer = csv.writer(result)
    file_writer.writerow(["Features Type"]+features_type_include)
    file_writer.writerow(["Features Name"]+list(features_name_include))
    for i in range(n_instances):
        file_writer.writerow([i]+list(dataset[i,:]))
    # Test whether there're any missing values
    for j in range(n_features-1):
        for i in range(n_instances):
            if dataset[i,j] == '':
                print "ERROR: STILL HAVE MISSING VALUES"
                print j
                break

