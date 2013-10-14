#------------------------------------------------------------------------------
# $Author: andybaoxv $
# $Date: 2012-11-26 02:42:53 -0500 (Mon, 26 Nov 2012) $
# $Revision: 198 $
#
# Northeastern University Machine Learning Group (NEUML)
#------------------------------------------------------------------------------

def parse_copdgene_data(data, features, case_ids, features_of_interest, \
                        cases_of_interest):
    """Parses COPDGene feature data

    This function can be used to parse COPDGene data. User specifies features
    of interest and cases of interest -- the function returns feature value
    data for these requests. The returned data will exclude entries for any
    cases that have missing data for at least one of the requested features.

    Parameters
    ----------
    data : array, shape ( n_instances, n_features )
        Matrix containing the feature values. Expected to be a superset of the
        requested data

    features : list, len ( n_features )
        List containing the names of the features, ordered according to the
        columns in 'data'. Expected to be a superset of requested features

    case_ids : list, len ( n_instances )
        This is a list containing the names of the cases, ordered according to
        the n rows in 'data'. Expected to be a superset of requested case_ids

    features_of_interest: list of strings
        List of features for which data is desired

    cases_of_interest: list of strings
        List of cases for which data is desired

    Returns
    -------
    data_of_interest : array, shape ( N, M )
        2D matrix of requested feature values for the requested cases. Here,
        'N' is the number of cases from 'cases_of_interest' that were
        actually found in the input data and that have no missing features

    found_cases_of_interest : list of strings
        List of cases from 'cases_of_interest' that were actually found in
        the input data. This list corresponds to the number of rows in
        'data_of_interest'.

    found_features_of_interest : list of strings
        List of features from 'features_of_interest' that were actually
        found in the input data. This list corresponds to the number of
        columns in 'data_of_interest'.

    missing_data_case_list : list of strings
        List of cases from 'cases_of_interest' that were found in the input
        data but for which at least one feature from 'features_of_interest'
        is missing in the input data

    missing_case_list : list of string
        List of cases from 'cases_of_interest' that were not found in the
        input data.

    missing_data_features_list : list of strings
        List of features from 'features_of_interest' that were found in the
        input data but for which at least one case from 'cases_of_interest'
        has a missing value

    missing_features_list : list of strings
        List of features from 'features_of_interest' that were not found in
        the input data.
    """
    
    import numpy as np
    
    # Output variable initialization
    found_cases_of_interest = []
    found_features_of_interest = []
    missing_data_cases_list = []
    missing_cases_list = []
    missing_data_features_list = []
    missing_features_list = []
    
    # num_row is the number of rows of the 'data'
    num_row = len(case_ids)
    
    # num_col is the number of columns of the 'data'
    num_col = len(features)
    
    # traverse through the 'case_ids' once
    for row in range(0,num_row):
        
        # if case_ids[row] is one of the cases_of_interest
        if case_ids[row] in cases_of_interest:
            
            # flag indicates whether case_id[row] has missing features
            flag = False
                        
            for j in features_of_interest:
                
                # flag_2 indicates whether j is in features
                flag_2 = False
                
                for jj in range(0,num_col):
                    if features[jj] == j:
                        flag_2 = True
                        index = jj
                        break
                if flag_2 == True:
                    if data[row][index] == 'NA':
                        flag = True
                        break
                        
            if flag == True:
                missing_data_cases_list.append(case_ids[row])
            else:
                found_cases_of_interest.append(case_ids[row])
    
    for case in cases_of_interest:
        if not(case in found_cases_of_interest):
            if not(case in missing_data_cases_list):
                missing_cases_list.append(case)
    
    # process features            
    for col in range(0,num_col):
        
        if features[col] in features_of_interest:
            
            flag = False
            
            for i in cases_of_interest:
                
                flag_2 = False
                for ii in range(0,num_row):
                    if case_ids[ii] == i:
                        flag_2 = True
                        index = ii
                        break
                
                if flag_2 == True:
                    if data[index][col] == 'NA':
                        flag = True
                        break
                        
            if flag == True:
                missing_data_features_list.append(features[col])
            else:
                found_features_of_interest.append(features[col])
                
    for feature in features_of_interest:
        if not(feature in found_features_of_interest):
            if not(feature in missing_data_features_list):
                missing_features_list.append(feature)
    
    num_row_1 = len(found_cases_of_interest)
    num_col_1 = len(found_features_of_interest)+len(missing_data_features_list)
    data_of_interest = np.zeros((num_row_1,num_col_1),dtype=float)
    ii = 0
    jj = 0
    for row in range(0,num_row):
        if case_ids[row] in found_cases_of_interest:
            for col in range(0,num_col):          
                if (features[col] in found_features_of_interest) or\
                (features[col] in missing_data_features_list):
                    data_of_interest[ii,jj] = float(data[row,col])
                    jj = jj+1
            ii = ii+1
            jj = 0

    return data_of_interest,\
           found_cases_of_interest,\
           found_features_of_interest,\
           missing_data_cases_list,\
           missing_cases_list,\
           missing_data_features_list,\
           missing_features_list

def find_data_through_feature(data,features,feature_interest):
    """
    Find data through a feature:
    This function gets all the values of one feature, which is specified by the
    input feature_interest, from an COPDgene data array, whcih is specified by 
    the input data
    
    Parameters
    ----------
    data : array, shape ( n_instances, n_features )
        Matrix containing the feature values. Expected to be a superset of the
        requested data

    features : list, len ( n_features )
        List containing the names of the features, ordered according to the
        columns in 'data'. Expected to be a superset of requested features
    
    feature_interest: list, len( 1 )
        List of one single features for which data is desired
    
    Returns
    ------
    result: List containing the data corresponding to feature_interest for all
         case_ids

    """
    result = []
    num_col = len(features)
    num_row = len(data[:,0])
    for col in range(0,num_col):
        if feature_interest == features[col]:
            for row in range(0,num_row):
                if data[row][col] != 'NA':
                    result.append(float(data[row][col]))
            break
    return result