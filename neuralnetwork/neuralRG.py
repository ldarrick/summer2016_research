#!/usr/bin/env python

#
# Last modified: 2 Aug 2016
# Author: Darrick Lee <y.l.darrick@gmail.com>
# 
# This script builds a neural network to try and classify red and green cells.

import csv
import numpy as NP
import tensorflow as TF
import random
from optparse import OptionParser

inputFile = ''

# 0 for MNIST style (NOTE: THIS CURRENTLY DOES NOT WORK)
# 1 for iris style
code_type = 1 

train_p = 0.99
valid_p = 0
learning_rate = 0.005
num_training_steps = 10
batch_size = 20

# Select features to use for neural network
featureNames = ['AreaShape_Compactness', 'AreaShape_Eccentricity', 
	'AreaShape_MajorAxisLength', 'AreaShape_MinorAxisLength',
	'AreaShape_Orientation', 'AreaShape_Perimeter',
	'AreaShape_Area']
featureNames = ['AreaShape_Area','AreaShape_Perimeter','AreaShape_MajorAxisLength','AreaShape_MinorAxisLength']
labelName = 'Label_RG'
numFeat = len(featureNames)

# Options Parsing
parser = OptionParser()
parser.add_option("-i", "--input", action="store", type="string", dest="inputFile", help="path to input csv file", metavar="INPUTCSV")

# Options parsing
(options, args) = parser.parse_args()
if options.inputFile:
	inputFile = options.inputFile

# Function to randomly split the data up
# train_p is the percentage of data points to use as training data
# valid_p is the percentage of data points to use as validation data
# 1 - train_p - valid_p is the percentage of data points to use as test data
def splitData(featList, labelList1, train_p, valid_p):
	numData = len(labelList1)

	# Randomly shuffle the data
	perm = NP.random.permutation(numData)
	featList = featList[perm,:]
	labelList1 = labelList1[perm][:]

	# Number of data points in each data set
	train_num = int(round(numData*train_p))
	valid_num = int(round(numData*valid_p))

	# Split up data sets
	feat_train = featList[0:train_num,:]
	label_train = labelList1[0:train_num][:]

	feat_valid = featList[train_num:train_num+valid_num,:]
	label_valid = labelList1[train_num:train_num+valid_num][:]

	feat_test = featList[train_num+valid_num:,:]
	label_test = labelList1[train_num+valid_num:][:]

	return feat_train, label_train, feat_valid, label_valid, feat_test, label_test


# Function to create a batch of data sets
def nextBatch(featList, labelList1, batchsize):
	numData = len(labelList1)

	# Randomly shuffle data
	perm = NP.random.permutation(numData)
	featList = featList[perm,:]
	labelList1 = labelList1[perm,:]

	feat_batch = featList[:batchsize,:]
	label_batch = labelList1[:batchsize,:]

	return feat_batch, label_batch

# Begin parsing the CSV file and build feature list
featList = []
labelList1 = [] # For MNIST style code
labelList2 = [] # For iris style code

with open(inputFile) as inputCSV:
	inputReader = csv.DictReader(inputCSV)

	# Go through each row and add relevant features
	for row in inputReader:
		# Add features
		cur_feat = []
		for feat in featureNames:
			cur_feat.append(float(row[feat]))

		# Add label
		if int(row[labelName]) == 0:
			labelList1.append([1,0])
			labelList2.append(0)
		else:
			labelList1.append([0,1])
			labelList2.append(1)

		featList.append(cur_feat)

featList = NP.array(featList)
labelList1 = NP.array(labelList1)
labelList2 = NP.array(labelList2)

# MNIST style
## NOTE: THIS CODE IS CURRENTLY NOT WORKING
if code_type == 0:
	# Split data up into training, validation and test data
	feat_train, label_train, feat_valid, label_valid, feat_test, label_test = splitData(featList, labelList1, train_p, valid_p)

	# Begin building the required structure for the neural network
	x = TF.placeholder(TF.float32, [None, numFeat])
	W = TF.Variable(TF.zeros([numFeat, 2]))
	b = TF.Variable(TF.zeros([2]))

	y = TF.nn.softmax(TF.matmul(x, W) + b)
	y_ = TF.placeholder(TF.float32, [None, 2])

	cross_entropy = TF.reduce_mean(-TF.reduce_sum(y_ * TF.log(y + 1e-50), reduction_indices=[1]))
	train_step = TF.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
	init = TF.initialize_all_variables()

	# Begin running the session
	sess = TF.Session()
	sess.run(init)

	for i in range(num_training_steps):
		batch_xs, batch_ys = nextBatch(feat_train, label_train, batch_size)
		print(batch_xs)
		if i == 1:
			print(sess.run(cross_entropy,feed_dict={x: batch_xs, y_: batch_ys}))

		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	# Check accuracy
	correct_prediction = TF.equal(TF.argmax(y,1), TF.argmax(y_,1))
	accuracy = TF.reduce_mean(TF.cast(correct_prediction, TF.float32))

	print(NP.shape(batch_xs))
	print(sess.run(TF.matmul(x, W),feed_dict={x: batch_xs, y_: batch_ys}))

	print(sess.run(TF.cast(b,TF.float32), feed_dict={x: feat_test, y_: label_test}))

elif code_type == 1:
	# Split data up into training, validation and test data
	feat_train, label_train, feat_valid, label_valid, feat_test, label_test = splitData(featList, labelList2, train_p, valid_p)

	classifier = TF.contrib.learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2)
	classifier.fit(x=feat_train, y=label_train, steps=100)

	# Evaluate accuracy.
	accuracy_score = classifier.evaluate(x=feat_test, y=label_test)["accuracy"]
	print('Accuracy: {0:f}'.format(accuracy_score))

	# Try a new sample
	new_samples = feat_test[:20,:]
	y = classifier.predict(new_samples)
	print ('Predictions: {}'.format(str(y)))
	print ('Real: {}'.format(str(label_test[:20])))