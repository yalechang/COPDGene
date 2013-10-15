"""Please referto <A Survey of Binary Similarity and Distance Measures> by
Seung-Seok Choi et.al
www.iiisci.org/journal/CV$/sci/pdfs/GS315JG.pdf
"""
import numpy as np
def similarity_binary(x,y,flag='sokal&michener'):
    """Compute similarity between binary vectors

    Parameters
    ----------
    x: vector, len(n)
        binary-valued vector
    y: vector, len(n)
        binary-valued vector, note that len(x) should be equal to len(y)
    flag: str
        choose which similarity metric to use
    Returns
    -------
    s: float
        similarity between these two input vectors using metric specified
    """
    
    # The length of two binary-valued vector should be equal
    assert len(x) == len(y)
    x_inv = [0]*len(x)
    y_inv = [0]*len(y)
    for i in range(len(x)):
        x_inv[i] = 1-x[i]
        y_inv[i] = 1-y[i]
    
    # Operational Taxonomic Units
    a = np.dot(x,y)
    b = np.dot(x_inv,y)
    c = np.dot(x,y_inv)
    d = np.dot(x_inv,y_inv)

    if flag == 'jaccard':
        s = a*1./(a+b+c)
    elif flag == 'dice':
        s = a*2./(2.*a+b+c)
    elif flag == 'czekanowski':
        s = a*2./(2.*a+b+c)
    elif flag == '3w-jaccard':
        s = a*3./(3.*a+b+c)
    elif flag == 'nei&li':
        s = a*2./((a+b)+(a+c))
    elif flag == 'sokal&sneath-I':
        s = a*1./(a+2*b+2*c)
    elif flag == 'sokal&michener':
        s = (a+d)*1./(a+b+c+d)
    else:
        s = (a+d)*2./(2.*a+b+c+2.*d)

    return s

def similarity_matrix_binary(data,flag='sokal&michener'):
    """Construct similarity matrix using a binary-valued dataset
    We need to compute similarity between each two features
    
    Parameters
    ----------
    data: array, shape(n_instances,n_features)
        original binary-valued dataset
    Returns
    -------
    mtr_similarity: array, shape(n_features,n_features)
        mtr_similarity[i,j] denotes the similarity value between feature i
        and feature j
    """
    n_instance,n_features = data.shape
    mtr_similarity = np.zeros((n_features,n_features))
    for i in range(n_features):
        for j in range(n_features):
            mtr_similarity[i,j] = similarity_binary(data[:,i],data[:,j])
    
    return mtr_similarity 

if __name__ == "__main__":
    x = [0,1,0,1]
    y = [1,1,0,1]
    #print similarity_binary(x,y)
