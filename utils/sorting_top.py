def sorting_top(seq,n_top,flag='min'):
    """This function find the values and indices of n_top elements in a
    sequence

    Parameters
    ----------
    seq: list
        sequence containing float values

    n_top: int
        the number of top values

    flag: str
        'min' means finding top smallest values
        'max' means finding top largest values

    Returns
    -------
    values: list,len(n_top)
        n_top values

    indices: list,len(n_top)
        indices in seq of elements in values
    """
    flag_active = [True]*len(seq)
    index = 0
    values = []
    indices = []

    while index<n_top:
        for i in range(len(seq)):
            if flag_active[i] == True:
                temp_val = seq[i]
                temp_index = i
                break

        for i in range(len(seq)):
            if flag_active[i] == True and seq[i]<temp_val:
                temp_val = seq[i]
                temp_index = i
        values.append(temp_val)
        indices.append(temp_index)
        flag_active[temp_index] = False
        index += 1
    
    return values,indices

def test_sorting_top():
    seq = [3,5,62,6,2,4,3,156,2,4,5,25,23,46,67]
    n_top = len(seq)
    print sorting_top(seq,n_top)

if __name__ == "__main__":
    test_sorting_top()
