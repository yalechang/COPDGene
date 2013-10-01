import numpy as np
import copy

def obtain_rank_use_score(features_name, features_importance):
    """This function obtains features rank according to features importance
    score. Note that the feature with the highest score should be rank 0 and
    the feature with lowest score should have rank value n_features-1

    Parameters
    ----------
    features_importance: list, len(n_features)
        features importance score for all the features

    Returns
    -------
    features_rank: list, len(n_features)
        features rank obtained by sorting the score in descending order
    features_importance_ranked: list, len(n_features)
        features importance score in descending order
    """
    features_name_ranked = copy.copy(features_name)
    features_importance_ranked = copy.copy(features_importance)
    n_features = len(features_importance)
    features_rank = range(n_features)
    
    # Rank features importance score in descending order
    for i in range(n_features-1):
        for j in range(i+1,n_features):
            if features_importance_ranked[i] < features_importance_ranked[j]:
                temp = features_importance_ranked[i]
                features_importance_ranked[i] = features_importance_ranked[j]
                features_importance_ranked[j] = temp
                tp = features_name_ranked[i]
                features_name_ranked[i] = features_name_ranked[j]
                features_name_ranked[j] = tp
    
    # Obtain the feature rank using feature names
    for i in range(n_features):
        for j in range(n_features):
            if features_name[i] == features_name_ranked[j]:
                features_rank[i] = j

    return features_rank,features_importance_ranked


if __name__ == "__main__":
    features_importance = [0.5,0.8,0.6,0.2,0.3]
    features_name = ['a','b','c','d','e']
    features_rank, features_importance_ranked = \
            obtain_rank_use_score(features_name,features_importance)

