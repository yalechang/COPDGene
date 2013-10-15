def parse_string(data,delimiter):
    """Parse a string where different small strings are separated by delimiter
    and obtain a list containing those small strings

    Parameters
    ----------
    data: string
        The original string that need to be parsed

    delimiter: char
        the char that separates the input string specified by data

    Returns
    -------
    str_list: list
        The list obtained through parsing the original string 'data'
    """
    
    #Initialize the return variable
    str_list = []
    str_num = 0
    #The length of original string
    data_len = len(data)
    
    temp = ""
    for i in range(data_len):
        if data[i] == delimiter:
            str_list.append(temp)
            str_num = str_num+1
            temp = ""
        else:
            temp = temp+data[i]
            if i == data_len-1 and data[i] != delimiter:
                str_list.append(temp)
    
    return str_list

if __name__ == "__main__":
    data = "whant\tdo\t\tyou\tlike\t?"
    print parse_string(data,'\t')

