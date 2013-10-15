def is_number(s):
    """
    Determine whether the content of a given input string is a number
    Motivation: For a dataset containing data of different types,such as int,
    float, str. It's better to store the whole dataset using an array whose
    elements are all str. Then to use the numerical values, it's necessary to
    make the tranforamtion.

    Parameters
    ----------
    s: str
        input string, need to determine if the content is a number

    Returns
    -------
    True/False
    """
    try:
        float(s)
        return True
    except ValueError:
        return False

if __name__ == "__main__":
    s = ['109','0.53','-10.9','+1.4']
    for i in range(len(s)):
        print is_number(s[i])
