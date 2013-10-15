import math
from is_number import is_number

def check_features_type(data):
    """Check the data types of different features

    Parameters
    ----------
    data: array,shape(n_instances,n_features)
        dataset

    Returns
    -------
    features_type: list, len(n_features)
        list containig the type of each feature
    """
    n_instances,n_features = data.shape
    features_type = [0]*n_features	
    
    # categorical features that are annotated
	# note that features should be identified by names instead of varnum because
	# we might use this for subsets of the whole dataset
    features_categorical = ['ccenter','FinalApproval','race',\
            'LimitWalkMost','StopWork']
    # ordinal features that are annotated
    features_ordinal = ['HealthStatus','HowSoonSmoke','SchoolCompleted']
    #print "======================================================================"
    for j in range(n_features):
        flag_categorical = True
        flag_interval = True
        flag_binary = True
        flag_continuous = True
        flag_ordinal = True
	    
        # Categorical feature
        if data[0,j] in features_categorical:
            flag_interval = False
            flag_binary = False
            flag_continuous = False
            flag_ordinal = False
        # Ordinal feature
        elif data[0,j] in features_ordinal:
            flag_categorical = False
            flag_interval = False
            flag_binary = False
            flag_continuous = False
	
	    # Feature types that are not specified manually
        else:
	        for i in range(1,n_instances): 
	            # If value is not a number, then classify it into 'categorical' type
	            # Here we assume any char type feature is categorical, which might not
	            # be true
	            if data[i,j] != '' and is_number(data[i,j]) == False:
	                flag_interval = False
	                flag_binary = False
	                flag_continuous = False
	                flag_ordinal = False
	            # If value is a number, consider both discrete and continuous
	            elif data[i,j] != '' and is_number(data[i,j]) == True:
	                flag_categorical = False
	                # Note that ordinal features could only be specified manually
	                flag_ordinal = False
	
	                # Convert str to number
	                temp = float(data[i,j])
	
	                # a feature is binary if values of this feature fall into [1,2,3]
	                if temp not in [0,1,2,3]:
	                    flag_binary = False
	
	                    # a feature is not interval
	                    if temp-math.floor(temp) != 0:
	                        flag_interval = False
        if flag_categorical == True:
	        features_type[j] = 'categorical'
        elif flag_ordinal == True:
	        features_type[j] = 'ordinal'
        elif flag_binary == True:
	        features_type[j] = 'binary'
        elif flag_interval == True and flag_binary == False:
	        features_type[j] = 'interval'
        elif flag_continuous == True:
	        features_type[j] = 'continuous'
    return features_type

if __name__ == "__main__":
    import numpy as np
    data = np.array([['str','int','float','binary'],
                     ['you','10','5.7','1'],
                     ['can','20','8.6','0'],
                     ['pass','30','4.3','1']])
    print check_features_type(data)
		
	
