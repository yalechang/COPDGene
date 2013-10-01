import numpy as np

def remove_redundancy(features_rank, thd, mtr_nhsic):
    """ This function remove redundancy between features according to method 2,
    ie. First we rank the features according to some criteria(dependency on
    GOLD/feature importance scores in random forest). Then we start from the
    rank 1 feature and keep one feature if and only if its dependency with all
    the selected features is no more than certain threshold thd.

    Parameters
    ----------
    features_rank: list, len(n_features)
        rank value for each feature, note that for convenience the range of 
        rank should be between 0 and n_features
    
    thd: float
        threshold value. A feature will be selected if and only if its 
        depdendency with features in the selected set is no more than this
        threshold

    mtr_nhsic: array, shape(n_features, n_features)
        matrix containing the dependency value between pairwise features. The
        dependency metric is Normalized HSIC

    Returns
    -------
    features_sel: list
        the ids of selected features. The length of this list should be
        determined by the input
    """
    n_features = len(features_rank)

    # Order feature according to ranking
    # features_ranked is the ids of ranked features, so it's a permutation of
    # the original features id, ie, (0,n_features)
    features_ranked = range(n_features)
    for i in range(n_features):
        features_ranked[features_rank[i]] = i

    # Select features
    # First select the feature with the highest ranking
    features_sel = [features_ranked[0]]
    for i in range(1,n_features):
        flag = True
        for j in range(len(features_sel)):
            if mtr_nhsic[features_ranked[i],features_sel[j]] > thd:
                flag = False
                break
        if flag == True:
            features_sel.append(features_ranked[i])

    return features_sel
