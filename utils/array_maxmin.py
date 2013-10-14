import numpy as np

def array_maxmin(s,flag='max'):
    
    n_samples,n_features = s.shape
    max_index = [0,0]
    min_index = [0,0]
    max_val = 0
    min_val = 0

    for i in range(n_samples):
        for j in range(n_features):
            if abs(s[i,j])>abs(max_val):
                max_val = s[i,j]
                max_index = [i,j]
            if abs(s[i,j])<abs(min_val):
                min_val = s[i,j]
                min_index = [i,j]

    if flag == 'max':
        return max_val
    elif flag == 'min':
        return min_val
    else:
        print "wrong input for array_maxmin"
        return 0


if __name__ == "__main__":
    s = np.arange(16).reshape(4,4)
    print array_maxmin(s,flag='max')
    print array_maxmin(s,flag='min')

