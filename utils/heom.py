def heom(data,features_type,val_max_col,val_min_col,m,n):
    from python.COPDGene.utils.is_number import is_number
    """This function computes Heterogeneous Euclidean Overlap Metric distance
    between m-th sample and n-th sample in a given dataset

    Parameters
    ----------
    data: array, shape(n_instances,n_features)
        array containing the original dataset
    
    features_type: list, len(n_features)
        list containing types of all the features, the values should fall into
        the set ['binary','categorical','interval','continuous']
    
    val_max_col: float
        max value for each column

    val_min_col: float
        min value for each column
    
    m: int
        row number of the first sample in the dataset

    n: int
        row number of the second sample in the dataset

    Returns
    -------
    dist_heom: float
        HEOM distance between i-th sample and k-th sample in dataset specified
        by data
    """
    n_instances,n_features = data.shape
    dist_temp = range(n_features)
    dist_sum = 0
    for j in range(n_features):
        if data[m,j] == '' or data[n,j] == '':
            dist_temp[j] = 1.
        elif features_type[j] in ['binary','categorical']:
            if data[m,j] == data[n,j]:
                dist_temp[j] = 0.
            else:
                dist_temp[j] = 1.
        elif features_type[j] in ['interval','continuous']:
            dist_temp[j] = (float(data[m,j])-float(data[n,j]))/\
                    (val_max_col[j]-val_min_col[j])
        else:
            pass
        dist_sum += dist_temp[j]**2
    dist_heom = dist_sum**0.5
    
    return dist_heom

def test_heom():
    import numpy as np
    data = np.array([[10,1],
                     [5,0],
                     [3,1],
                     [1,0]])
    features_type = ['interval','binary']
    val_max_col = [10,0]
    val_min_col = [1,0]
    print data
    print heom(data,features_type,val_max_col,val_min_col,0,1)
    print heom(data,features_type,val_max_col,val_min_col,0,2)
    print heom(data,features_type,val_max_col,val_min_col,0,3)

def heom_array(data,features_type):
    """Compute heom distance between pairwise samples in data

    Parameters
    ----------
    data: array,shape(n_instances,n_features)
        array containing the dataset

    features_type: list, len(n_features)
        types of every feature

    Returns
    -------
    mtr_heom: array, shape(n_instances,n_instances)
        array containing pairwise heom distance
    """
    import numpy as np
    n_instances,n_features = data.shape
    mtr_heom = np.zeros((n_instances,n_instances))

    val_max_col = [0]*n_features
    val_min_col = [0]*n_features
    for j in range(n_features):
        if features_type[j] in ['interval','continuous']:
            for i in range(n_instances):
                if data[i,j]!= '':
                    val_max_col[j] = float(data[i,j])
                    val_min_col[j] = float(data[i,j])
                    break
            for i in range(n_instances):
                if data[i,j]!='':
                    if float(data[i,j])>val_max_col[j]:
                        val_max_col[j] = float(data[i,j])
                    if float(data[i,j])<val_min_col[j]:
                        val_min_col[j] = float(data[i,j])
        
    for i in range(n_instances-1):
        for j in range(i+1,n_instances):
            mtr_heom[i,j] = heom(data,features_type,val_max_col,val_min_col,i,j)
            mtr_heom[j,i] = mtr_heom[i,j]
        print i 
    return mtr_heom

def test_heom_array():
    import numpy as np
    data = np.array([['10','1'],
                     ['5','0'],
                     ['3','1'],
                     ['1','0']])
    features_type = ['interval','binary']
    print heom_array(data,features_type)

if __name__ == "__main__":
    
    from time import time
    import pickle
    import csv

    t0 = time()
    data,features_name,features_type = \
            pickle.load(open("data_include.pkl","rb"))
    """
    mtr_heom = heom_array(data,features_type)
    n_instances,n_features = data.shape
    
    # Write the heom matrix into a csv file
    result = open("mtr_heom.csv","wb")
    file_writer = csv.writer(result)
    for i in range(n_instances):
        file_writer.writerow(mtr_heom[i,:])
    result.close()

    t1 = time()
    print(["Running Time(min)",(t1-t0)/60])
    """
    n_instances,n_features = data.shape
    n_missing = [0]*n_instances
    for i in range(n_instances):
        for j in range(n_features):
            if data[i,j] == '':
                n_missing[i] += 1
    print n_features
    print sum(n_missing)*1./n_instances
