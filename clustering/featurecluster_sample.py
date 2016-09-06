#
# Last modified: 6 September 2016
# Authors: Darrick Lee <y.l.darrick@gmail.com>
# Description: This is a test script to show how featurecluster works

import os
from featurecluster import FeatureCluster

## DEFINE PARAMETERS ##########################################################

# This is a single-frame feature extraction genreated using experimentalcsv.py
# This has the following columns:
#	Identifier (Cell ID) Column: 0
#	Location Columns: 1-2
#	Standardized Feature Columns: 3-12 (Use these for clustering)
#	Original Feature Columns: 13-22 (Use these for plotting)
#	Number of Tracked Cells: 565
load_file = 'test_f100_singleframe.csv'

# Define the FeatureCluster class
FC = FeatureCluster(load_file)

# Define the list of features to cluster over
# In this case, we want to cluster over all of the features
clist = range(3,13)

# Define the pairs of features to plot the clusters on
# Note that negative numbers correspond to the PCA components
# These pairs are:
#	(x-location, speed)
#	(area, perimeter)
#	(PCA1, PCA2)
plist = [[1,22], [14,15], [-1, -2]]

# Define where these plots should be stored
# Normally, we can specify different folders for each plot, but in this case we will use the same one
outputFolder = 'outputData/'

# Create the folder if it doesn't exist
if not os.path.isdir(outputFolder):
	os.mkdir(outputFolder)

pathlist = [outputFolder, outputFolder, outputFolder]

## PERFORM CLUSTERING #########################################################

# Perform agglomerative clustering
FC.AgglomerativeClustering(clist)

# Perform feature agglomeration
FC.FeatureAgglomeration(clist)

# Perform PCA analysis
FC.PCA(clist)

# Plot the agglomerative clusters over our specified pairs
FC.PlotAgglomerative(plist, pathlist)

# Create the feature tree
# NOTE: This requires graphviz, a separate graph visualization software
FC.PlotFeatureTree(outputFolder)

# Plot the PCA axes
FC.PlotPCA(outputFolder)