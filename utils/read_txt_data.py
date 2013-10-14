import numpy as np
from parse_string import parse_string

def read_txt_data(file_name):
    """This function reads txt format dataset and return an array containing
    all the information in the original txt file
    This function should be used in conjugate with the function parse_string

    Parameters
    ---------
    file_name: string
        the name of the file that need to be read into the program

    Returns
    -------
    mtr: array, shape(n_instances+1,n_features+1)
        the matrix containing all the information in the txt file
        Note that include the names of features
        Also include the IDs of patiences
    """
    txtfile = open(file_name)
    data = []
    for line in txtfile:
        data.append(line)

    # The number of instances contained in the data file
    n_instances = len(data)

    str_list = parse_string(data[0],'\t')
    n_features = len(str_list)

    # Initialize the array that need to be returned
    mtr = np.empty((n_instances,n_features),dtype=list)
    mtr[0,:] = str_list

    for i in range(1,n_instances):
        str_list = parse_string(data[i],'\t')
        mtr[i,:] = str_list
    txtfile.close()

    return mtr
