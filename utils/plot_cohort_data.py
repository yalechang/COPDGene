#------------------------------------------------------------------------------
# $Author: jross $
# $Date: 2013-01-31 11:54:49 -0500 (Thu, 31 Jan 2013) $
# $Revision: 302 $
#
# Northeastern University Machine Learning Group (NEUML)
#------------------------------------------------------------------------------

import numpy as np
from optparse import OptionParser
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from read_copdgene_data import read_copdgene_data
from parse_copdgene_data import parse_copdgene_data
from parse_copdgene_data import find_data_through_feature as fdtf

desc = """The script permits you to plot cohort data in a variety of ways \
for specified features. The csv file passed to this script is expected to \
have the first row indicate feature names and the first column indicate \
cohort SIDs."""

parser = OptionParser(description=desc)
parser.add_option('-f', '--file', \
                  help='Cohort data file name in csv format', \
                  dest='file_name', metavar='<file_name>')
parser.add_option('--hist_2d', help='Plot 2D histogram of specified features', \
                  dest='hist_2d_features', nargs=2, \
                  metavar='<feature_name1> <feature_name2>', default=[])
parser.add_option('--hist_1d', help='Plot 1D histogram of specified feature', \
                  dest='hist_1d_feature', nargs=1,
                  metavar='<feature_name>', default=[])
parser.add_option('--bar_3d', help='3D bar plot (2D histogram) of specified \
                  features', 
                  dest='bar_3d_features', nargs=2, \
                  metavar='<feature_name1> <feature_name2>', default=[])
parser.add_option('--scatter_2d', \
                  help='Plot 2D scatter plot of specified features', \
                  dest='scatter_2d_features', nargs=2,
                  metavar='<feature_name1> <feature_name2>', default=[])
parser.add_option('--scatter_3d', \
                  help='Plot 3D scatter plot of specified features', \
                  dest='scatter_3d_features', nargs=3,
                  metavar='<feature_name1> <feature_name2> <feature_name3>',
                 default=[])
parser.add_option('--num_bins_f1',
                  help='Number of histogram bins to use for first specified \
                  feature. Default is 30.',
                  dest='num_bins_feature1', nargs=1,
                  metavar='<num_bins_feature1>', type='int',
                  default=30)
parser.add_option('--num_bins_f2',
                  help='Number of histogram bins to use for second specified \
                  feature. Default is 30.',
                  dest='num_bins_feature2', nargs=1,
                  metavar='<num_bins_feature2>', type='int',
                  default=30)

(options, args) = parser.parse_args()

# Now read the data and parse according to the features specified by the user
[data,features,case_ids] = read_copdgene_data(options.file_name)

# Check to see if the user has specified a feature to plot a 1D histogram of
# and plot if so
if len(options.hist_1d_feature) > 0:
    [data_of_interest, found_cases_of_interest, found_features_of_interest,
     missing_data_cases_list, missing_cases_list, missing_data_features_list,
     missing_features_list] = parse_copdgene_data(data, features, case_ids,
                                                [options.hist_1d_feature],
                                                case_ids)
    if len(missing_features_list) == 0:
        feature_1 = list(data_of_interest[:,0])        
        plt.figure()
        plt.hist(feature_1, options.num_bins_feature1)
        plt.xlabel(options.hist_1d_feature)
        plt.show()
    else:
        print "The input feature '",options.hist_1d_feature,"' does not exist\
        in the csv file you have specified\nPlease input a different feature\
        name"

# Check to see if the user has specified features to plot a 2D histogram of
# and plot if so
if len(options.hist_2d_features) == 2:
    [data_of_interest, found_cases_of_interest, found_features_of_interest,
     missing_data_cases_list, missing_cases_list, missing_data_features_list,
     missing_features_list] = parse_copdgene_data(data, features, case_ids,
                                                options.hist_2d_features,
                                                case_ids)
    if len(missing_features_list) == 0:
        feature_1 = list(data_of_interest[:,0])
        feature_2 = list(data_of_interest[:,1])       
        H, xedges, yedges = np.histogram2d(feature_1, feature_2,
                                           bins=[options.num_bins_feature2,
                                                 options.num_bins_feature1],
                                           normed=True)

        extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
        if len(found_features_of_interest) == 2:
            if set(feature_1)<=set(fdtf(data,features,\
            found_features_of_interest[0])):
                plt.xlabel(found_features_of_interest[0])
                plt.ylabel(found_features_of_interest[1])
            else:
                plt.xlabel(found_features_of_interest[1])
                plt.ylabel(found_features_of_interest[0])
        elif len(found_features_of_interest) == 1:
            if set(feature_1)<=set(fdtf(data,features,\
            found_features_of_interest[0])):
                plt.xlabel(found_features_of_interest[0])
                plt.ylabel(missing_data_features_list[0])
            else:
                plt.xlabel(missing_data_features_list[0])
                plt.ylabel(found_features_of_interest[0])
        else:
            if set(feature_1)<=set(fdtf(data,features,\
            missing_data_features_list[0])):
                plt.xlabel(missing_data_features_list[0])
                plt.ylabel(missing_data_features_list[1])
            else:
                plt.xlabel(missing_data_features_list[1])
                plt.ylabel(missing_data_features_list[0])
        plt.imshow(H,aspect='auto',extent=extent,interpolation='nearest')
        plt.colorbar()
        plt.show()
    elif len(missing_features_list) == 1:
        print "The following features requested do not exist:"
        print missing_features_list[0]
    else:
        print "The following features requested do not exist:"
        print missing_features_list[0]
        print missing_features_list[1]

# Check to see if the user has specified features for a 3D bar plot and plot if
# so
if len(options.bar_3d_features) == 2:
    [data_of_interest, found_cases_of_interest, found_features_of_interest,
     missing_data_cases_list, missing_cases_list, missing_data_features_list,
     missing_features_list] = parse_copdgene_data(data, features, case_ids,
                                                options.bar_3d_features,
                                                case_ids)
    if len(missing_features_list) == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        feature_1 = list(data_of_interest[:,0])
        feature_2 = list(data_of_interest[:,1])

        hist, xedges, yedges = np.histogram2d(feature_1, feature_2,
                                              bins=[options.num_bins_feature2,
                                                    options.num_bins_feature1],
                                              normed=False)
        elements = (len(xedges) - 1)*(len(yedges) - 1)
        xpos, ypos = np.meshgrid(xedges[:-1]+0.25, yedges[:-1]+0.25)
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros(elements)
        dx = (xedges[1] - xedges[0])*np.ones_like(zpos)
        dy = (yedges[1] - yedges[0])*np.ones_like(zpos)
        dz = hist.flatten()
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
        if len(found_features_of_interest) == 2:
            if set(feature_1)<=set(fdtf(data,features,\
            found_features_of_interest[0])):
                ax.set_xlabel(found_features_of_interest[0])
                ax.set_ylabel(found_features_of_interest[1])
            else:
                ax.set_xlabel(found_features_of_interest[1])
                ax.set_ylabel(found_features_of_interest[0])
        elif len(found_features_of_interest) == 1:
            if set(feature_1)<=set(fdtf(data,features,\
            found_features_of_interest[0])):
                ax.set_xlabel(found_features_of_interest[0])
                ax.set_ylabel(missing_data_features_list[0])
            else:
                ax.set_xlabel(missing_data_features_list[0])
                ax.set_ylabel(found_features_of_interest[0])
        else:
            if set(feature_1)<=set(fdtf(data,features,\
            missing_data_features_list[0])):
                ax.set_xlabel(missing_data_features_list[0])
                ax.set_ylabel(missing_data_features_list[1])
            else:
                ax.set_xlabel(missing_data_features_list[1])
                ax.set_ylabel(missing_data_features_list[0])
        plt.show()
    elif len(missing_features_list) == 1:
        print "The following features requested do not exist:"
        print missing_features_list[0]
    else:
        print "The following features requested do not exist:"
        print missing_features_list[0]
        print missing_features_list[1]

# Check to see if the user has specified features to plot a 2D scatter plot of,
# and plot if so
if len(options.scatter_2d_features) == 2:
    [data_of_interest, found_cases_of_interest, found_features_of_interest,
     missing_data_cases_list, missing_cases_list, missing_data_features_list,
     missing_features_list] = parse_copdgene_data(data, features, case_ids,
                                                options.scatter_2d_features,
                                                case_ids)
    if len(missing_features_list) == 0:
        plt.figure()
        feature_1 = list(data_of_interest[:,0])
        feature_2 = list(data_of_interest[:,1])
        if len(found_features_of_interest) == 2:
            if set(feature_1)<=set(fdtf(data,features,\
            found_features_of_interest[0])):
                plt.xlabel(found_features_of_interest[0])
                plt.ylabel(found_features_of_interest[1])
            else:
                plt.xlabel(found_features_of_interest[1])
                plt.ylabel(found_features_of_interest[0])
        elif len(found_features_of_interest) == 1:
            if set(feature_1)<=set(fdtf(data,features,\
            found_features_of_interest[0])):
                plt.xlabel(found_features_of_interest[0])
                plt.ylabel(missing_data_features_list[0])
            else:
                plt.xlabel(missing_data_features_list[0])
                plt.ylabel(found_features_of_interest[0])
        else:
            if set(feature_1)<=set(fdtf(data,features,\
            missing_data_features_list[0])):
                plt.xlabel(missing_data_features_list[0])
                plt.ylabel(missing_data_features_list[1])
            else:
                plt.xlabel(missing_data_features_list[1])
                plt.ylabel(missing_data_features_list[0])
        plt.scatter(feature_1, feature_2, c='b', marker='o')
        plt.show()
    elif len(missing_features_list) == 1:
        print "The following features requested do not exist:"
        print missing_features_list[0]
    else:
        print "The following features requested do not exist:"
        print missing_features_list[0]
        print missing_features_list[1]
                                              
# Check to see if the user has specified features to plot a 3D scatter plot of,
# and plot if so
if len(options.scatter_3d_features) == 3:
    [data_of_interest, found_cases_of_interest, found_features_of_interest,
     missing_data_cases_list, missing_cases_list, missing_data_features_list,
     missing_features_list] = parse_copdgene_data(data, features, case_ids,
                                                options.scatter_3d_features,
                                                case_ids)
    if len(missing_features_list) == 0:
        fig = plt.figure()
        feature_1 = list(data_of_interest[:,0])
        feature_2 = list(data_of_interest[:,1])
        feature_3 = list(data_of_interest[:,2])
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(feature_1, feature_2, feature_3, c='b')
        if len(found_features_of_interest) == 3:
            if set(feature_1)<=set(fdtf(data,features,\
            found_features_of_interest[0])):
                ax.set_xlabel(found_features_of_interest[0])
            elif set(feature_1) in set(fdtf(data,features,\
            found_features_of_interest[1])):
                ax.set_xlabel(found_features_of_interest[1])
            else:
                ax.set_xlabel(found_features_of_interest[2])
                
            if set(feature_2)<=set(fdtf(data,features,\
            found_features_of_interest[0])):
                ax.set_ylabel(found_features_of_interest[0])
            elif set(feature_2)<=set(fdtf(data,features,\
            found_features_of_interest[1])):
                ax.set_ylabel(found_features_of_interest[1])
            else:
                ax.set_ylabel(found_features_of_interest[2])
            
            if set(feature_3)<=set(fdtf(data,features,\
            found_features_of_interest[0])):
                ax.set_zlabel(found_features_of_interest[0])
            elif set(feature_3)<=set(fdtf(data,features,\
            found_features_of_interest[1])):
                ax.set_zlabel(found_features_of_interest[1])
            else:
                ax.set_zlabel(found_features_of_interest[2])
                
        elif len(found_features_of_interest) == 2:
            if set(feature_1)<=set(fdtf(data,features,\
            found_features_of_interest[0])):
                ax.set_xlabel(found_features_of_interest[0])
            elif set(feature_1)<=set(fdtf(data,features,\
            found_features_of_interest[1])):
                ax.set_xlabel(found_features_of_interest[1])
            else:
                ax.set_xlabel(missing_data_features_list[0])
                
            if set(feature_2)<=set(fdtf(data,features,\
            found_features_of_interest[0])):
                ax.set_ylabel(found_features_of_interest[0])
            elif set(feature_2)<=set(fdtf(data,features,\
            found_features_of_interest[1])):
                ax.set_ylabel(found_features_of_interest[1])
            else:
                ax.set_ylabel(missing_data_features_list[0])
            
            if set(feature_3)<=set(fdtf(data,features,\
            found_features_of_interest[0])):
                ax.set_zlabel(found_features_of_interest[0])
            elif set(feature_3)<=set(fdtf(data,features,\
            found_features_of_interest[1])):
                ax.set_zlabel(found_features_of_interest[1])
            else:
                ax.set_zlabel(missing_data_features_list[0])
                
        elif len(found_features_of_interest) == 1:
            if set(feature_1)<=set(fdtf(data,features,\
            found_features_of_interest[0])):
                ax.set_xlabel(found_features_of_interest[0])
            elif set(feature_1)<=set(fdtf(data,features,\
            missing_data_features_list[0])):
                ax.set_xlabel(missing_data_features_list[0])
            else:
                ax.set_xlabel(missing_data_features_list[1])
                
            if set(feature_2)<=set(fdtf(data,features,\
            found_features_of_interest[0])):
                ax.set_ylabel(found_features_of_interest[0])
            elif set(feature_2)<=set(fdtf(data,features,\
            missing_data_features_list[0])):
                ax.set_ylabel(missing_data_features_list[0])
            else:
                ax.set_ylabel(missing_data_features_list[1])
            
            if set(feature_3)<=set(fdtf(data,features,\
            found_features_of_interest[0])):
                ax.set_zlabel(found_features_of_interest[0])
            elif set(feature_3)<=set(fdtf(data,features,\
            missing_data_features_list[0])):
                ax.set_zlabel(missing_data_features_list[0])
            else:
                ax.set_zlabel(missing_data_features_list[1])
        else:
            if set(feature_1)<=set(fdtf(data,features,\
            missing_data_features_list[0])):
                ax.set_xlabel(missing_data_features_list[0])
            elif set(feature_1)<=set(fdtf(data,features,\
            missing_data_features_list[1])):
                ax.set_xlabel(missing_data_features_list[1])
            else:
                ax.set_xlabel(missing_data_features_list[2])
                
            if set(feature_2)<=set(fdtf(data,features,\
            missing_data_features_list[0])):
                ax.set_ylabel(missing_data_features_list[0])
            elif set(feature_2)<=set(fdtf(data,features,\
            missing_data_features_list[1])):
                ax.set_ylabel(missing_data_features_list[1])
            else:
                ax.set_ylabel(missing_data_features_list[2])
            
            if set(feature_3)<=set(fdtf(data,features,\
            missing_data_features_list[0])):
                ax.set_zlabel(missing_data_features_list[0])
            elif set(feature_3)<=set(fdtf(data,features,\
            missing_data_features_list[1])):
                ax.set_zlabel(missing_data_features_list[1])
            else:
                ax.set_zlabel(missing_data_features_list[2])
        plt.show()
    elif len(missing_features_list) == 1:
        print "The following features requested do not exist:"
        print missing_features_list[0]
    elif len(missing_features_list) == 2:
        print "The following features requested do not exist:"
        print missing_features_list[0]
        print missing_features_list[1]
    else:
        print "The following features requested do not exist:"
        print missing_features_list[0]
        print missing_features_list[1]
        print missing_features_list[2]
