#------------------------------------------------------------------------------
# $Author: andybaoxv $
# $Date: 2012-11-15 19:47:24 -0500 (Thu, 15 Nov 2012) $
# $Revision: 184 $
#
# Northeastern University Machine Learning Group (NEUML)
#------------------------------------------------------------------------------

def read_copdgene_data(file_name):
    """Reads COPDGene feature files (csv format)

    This function can be used to read COPDGene data. The data file is assumed
    to be a csv file in which the first column lists the study ID and the
    first row lists the feature names. The function reads feature values, case
    names, and feature names into separate containers.

    Parameters
    ----------
    file_name : string
        Name of COPDGene data file (assumed to be a csv file)

    Returns
    -------
    data : array, shape ( n_instances, n_features )
        Matrix containing the feature values

    features : list, len ( n_features )
        List containing the names of the features, ordered according to the
        columns in 'data'

    case_ids : list, len ( n_instances )
        This is a list containing the names of the cases, ordered according to
        the n rows in 'data'
    """
    import numpy as np
    import csv
    
    # 'case_ids' is a list that contains ID for every case in the csv data file
    case_ids = []

    # Every row contains all of one person's feature data,every column contains
    # data of one feature for all persons
    data = []

    # Open the csvfile in read-only mode
    csvfile = open(file_name,'rb') 
    reader  = csv.reader(csvfile)  
    
    # 'lines' contain a copy of all the data in the csv file
    lines = [line for line in reader]
    
    # Get the number of rows and columns of the csv file
    num_row = len(lines)
    num_col = len(lines[0])
    
    # Initialize the array 'data' 
    data = np.empty((num_row-1,num_col-1),dtype=list)
    
    # First row(except 1st element) of the the file lists the features
    features = lines[0]
    del features[0]

    # Loop through the file and populate 'data' and 'case_id'
    for k in range(1, num_row): 
        
        # The 1st element in the list is the ID of one case
        case_ids.append(lines[k][0])

        # The remaining elements in the list are features of the instance
        data[(k-1),] = lines[k][1:num_col]

    # Close the file after finishing reading all the data
    csvfile.close() 
    return data, features, case_ids
