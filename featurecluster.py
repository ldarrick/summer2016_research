#
# Last modified: 20 June 2016
# Authors: Darrick Lee <y.l.darrick@gmail.com>
# Description: The class definition which takes in a set of feature vectors in a csv file and can 
# perform various methods of clustering and analysis on it
#
# Note about variable names:
# 	clist refers to the list of features to cluster, based on column number
#		- Ex. clist = [4,5,6,7,8] will cluster features 4-8
#	plist refers to a list of 2-tuples that define which combination of axes to plot against
#		- Ex. plist = [[1,7],[8,1]] will make two plots.
#		  One will have feature 1 on x-axis and feature 7 on y-axis
#		  The other will have feature 8 on x-axis and feature 1 on y-axis
#	flist refers to a list of features to plot on the biplot for PCA
#
# Note about PCA variables:
#	PCA variables can be accessed in clist and plist by using negative integers
#	-1 refers to the first PCA variable
#	-2 refers to the second PCA variable, etc.
#
# Required Packages: scikit-learn

import numpy as NP
import matplotlib.pyplot as PLT

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
from sklearn.decomposition import PCA

class FeatureCluster:

	def __init__(self, featcsv):

		# Import data from the csv file
		self.featList = NP.genfromtxt(featcsv, delimiter=',', dtype='float')[1:]
		self.numCells = len(self.featList)
		self.featNames = NP.genfromtxt(featcsv, delimiter=',', skip_footer=self.numCells, dtype='str')

		# Agglomerative Variables
		self.agglomerativeLabel = None
		self.agglomerativeCList = None # The cluster list for agglomerative clustering
		self.agglomerativeCNum = None # Number of agglomerative clusters

		# K-Means Variables
		self.kmeansLabel = None
		self.kmeansCList = None # The cluster list for k means clustering
		self.kmeansCNum = None # Number of kmeans clusters

		# Feature Agglomeration Variables
		self.featureTree = None # The list of children of each node in the feature agglomeration tree
		self.featureLabels = None 

		# PCA Variables
		self.PCAfeatList = None # The feature list transformed into PCA variables
		self.PCAvariables = None # The variables used for PCA
		self.PCAcomponents = None # The components of the PCA variables
		self.PCAexplainedVariance = None # The explained variance of the PCA variables

		# Plotting Variables
		self.clusterDotSize = 12

	def AgglomerativeClustering(self, clist, numClusters=2):
		AGGL = AgglomerativeClustering(numClusters)
		self.agglomerativeLabel = AGGL.fit_predict(self.featList[:,clist])
		self.agglomerativeCNum = numClusters
		self.agglomerativeCList = clist


	def FeatureAgglomeration(self, clist, numClusters=2):
		FEATAGGL = FeatureAgglomeration(numClusters)
		FEATAGGL.fit(self.featList[:,clist])
		self.featureTree = FEATAGGL.children_	
		self.featureLabels = FEATAGGL.labels_

	def KMeansClustering(self, clist, numClusters=2):
		KMEANS = KMeansClustering(numClusters)
		self.kmeansLabel = KMEANS.fit_predict(self.featList[:,clist])
		self.kmeansCNum = numClusters
		self.kmeansCList = clist

	def PCA(self, clist, numComponents=2):
		PCA_MODEL = PCA(numComponents)
		self.PCAfeatList = PCA_MODEL.fit_transform(self.featList[:,clist])
		self.PCAcomponents = PCA_MODEL.components_
		self.PCAexplainedVariance = PCA_MODEL.explained_variance_ratio_
		self.PCAvariables = clist 

	def PlotAgglomerative(self, plist):
		numPlots = len(plist)
		colorspace = NP.linspace(0,1,self.agglomerativeCNum+1)[:-1]
		colors = PLT.cm.hsv(colorspace)

		for j, plotaxes in enumerate(plist):
			PLT.figure()

			# Create axes (check if PCA axes are requested)
			if plotaxes[0]<0:
				X1 = self.PCAfeatList[:,abs(plotaxes[0])-1]
				xlabel = 'PCA' + str(abs(plotaxes[0]))
			else:
				X1 = self.featList[:,plotaxes[0]]
				xlabel = self.featNames[plotaxes[0]]
			if plotaxes[1]<0:
				X2 = self.PCAfeatList[:,abs(plotaxes[1])-1]
				ylabel = 'PCA' + str(abs(plotaxes[1]))
			else:
				X2 = self.featList[:,plotaxes[1]]
				ylabel = self.featNames[plotaxes[1]]

			for i, c in enumerate(colors):
				PLT.scatter(X1[self.agglomerativeLabel==i], X2[self.agglomerativeLabel==i], color=c,
					edgecolor='black', s=self.clusterDotSize)

			PLT.xlabel(xlabel)
			PLT.ylabel(ylabel)
			PLT.savefig('Agglomerative' + str(j), bbox_inches='tight', dpi=400)


	def PlotKMeans(self, plist):
		numPlots = len(plist)
		colorspace = NP.linspace(0,1,self.kmeansCNum+1)[:-1]
		colors = PLT.cm.hsv(colorspace)

		for j, plotaxes in enumerate(plist):
			PLT.figure()

			# Create axes (check if PCA axes are requested)
			if plotaxes[0]<0:
				X1 = self.PCAfeatList[:,abs(plotaxes[0])-1]
				xlabel = 'PCA' + str(abs(plotaxes[0]))
			else:
				X1 = self.featList[:,plotaxes[0]]
				xlabel = self.featNames[plotaxes[0]]
			if plotaxes[1]<0:
				X2 = self.PCAfeatList[:,abs(plotaxes[1])-1]
				ylabel = 'PCA' + str(abs(plotaxes[1]))
			else:
				X2 = self.featList[:,plotaxes[1]]
				ylabel = self.featNames[plotaxes[1]]

			for i, c in enumerate(colors):
				PLT.scatter(X1[self.kmeansLabel==i], X2[self.kmeansLabel==i], color=c,
					edgecolor='black', s=self.clusterDotSize)

			PLT.xlabel(xlabel)
			PLT.ylabel(ylabel)
			PLT.savefig('KMeans' + str(j), bbox_inches='tight', dpi=400)

	def PlotPCA(self, flist):
		sc = 2
		biplotList = []
		features = []

		# Pick out the features that are actually used in PCA
		for f in flist:
			if f in self.PCAvariables:
				biplotList.append(self.PCAvariables.index(f))
				features.append(self.featNames[f])

		xvector = self.PCAcomponents[0][biplotList]
		yvector = self.PCAcomponents[1][biplotList]

		for i in range(len(xvector)):
			PLT.arrow(0, 0, xvector[i]*sc, yvector[i]*sc,
			          color='g', width=0.005, head_width=0.05)
			PLT.text(xvector[i]*sc*1.1, yvector[i]*sc*1.1,
			         features[i], color='g')

		axes = PLT.gca()
		axes.set_xlim([-1,1])
		axes.set_ylim([-1,1])

		PLT.savefig('PCABiPlot', bbox_inches='tight', dpi=400)

