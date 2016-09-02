#
# Last modified: 22 August 2016
# Authors: Darrick Lee <y.l.darrick@gmail.com>
# Description: The class definition which builds a convolutional neural network (CNN) to classify images.
# The structure of the CNN is:
#	- A variable number of convolutional/pooling layers (user defined)
#	- 1 fully connected layer
#	- 1 fully connected softmax output layer
#
# Based on MNIST CNN Tensorflow Tutorial: https://www.tensorflow.org/versions/r0.10/tutorials/mnist/pros/index.html
#
# Required Packages: tensorflow


import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

## HELPER FUNCTIONS ###########################################################
# Initialize weight variable to Gaussian random variable
# Default standard deviation: 0.1
def weight_variable(shape, sigma=0.1):
	initial = tf.truncated_normal(shape, stddev=sigma)
	return tf.Variable(initial)

# Initialize bias variable to a constant vector
# Default constant = 0.1
def bias_variable(shape, c=0.1):
	initial = tf.constant(c, shape=shape)
	return tf.Variable(initial)

# Convolution function
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Max pooling function 
def max_pool(x, n=2):
	return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')

# Function to randomly split the data up
# train_p is the percentage of data points to use as training data
# valid_p is the percentage of data points to use as validation data
# 1 - train_p - valid_p is the percentage of data points to use as test data
def splitData(featList, labelList, train_p, valid_p):
	numData = np.shape(featList)[0]

	# Randomly shuffle the data
	perm = np.random.permutation(numData)
	featList = featList[perm][:][:]
	labelList = labelList[perm][:]

	# Number of data points in each data set
	train_num = int(round(numData*train_p))
	valid_num = int(round(numData*valid_p))

	# Split up data sets
	feat_train = featList[0:train_num][:][:]
	label_train = labelList[0:train_num][:]

	feat_valid = featList[train_num:train_num+valid_num][:][:]
	label_valid = labelList[train_num:train_num+valid_num][:]

	feat_test = featList[train_num+valid_num:][:][:]
	label_test = labelList[train_num+valid_num:][:]

	return feat_train, label_train, feat_valid, label_valid, feat_test, label_test

# Function to create a random batch of data sets
def nextBatch(featList, labelList, batchsize):
	numData = np.shape(featList)[0]

	# Randomly shuffle data
	perm = np.random.permutation(numData)
	featList = featList[perm][:][:]
	labelList = labelList[perm][:]

	feat_batch = featList[:batchsize][:][:]
	label_batch = labelList[:batchsize][:]

	return feat_batch, label_batch

## CLASS DEFINITION ###############################################################
class CNNClassifier:

	## CONSTRUCTOR ################################################################
	# Input arguments:
	#	imgSize: The size of one dimension of the image (we assume that the image is square)
	#	numClass: The number of output classes
	#	fmaps: A list of numbers of feature maps per layer (length of list is number of layers)
	#	convsize: A list of convolutional filter size per layer (length of list is number of layers)
	#	poolsize: A list of the pooling size per layer (length of list is number of layers)
	#	fcsize: Size of fully connected layer at the end
	def __init__(self, imgSize, numClass, fmaps, convsize, poolsize, fcsize):

		# Initialize data variables
		self.images_full = None # Full set of input images
		self.images_train = None # Training images
		self.images_valid = None # Validation images
		self.images_test = None # Test images

		self.labels_full = None # Full set of input labels
		self.labels_train = None # Training labels
		self.labels_valid = None # Validation labels
		self.labels_test = None # Test labels

		# Save all parameters as local variables
		self.imgSize = imgSize
		self.numClass = numClass

		# Begin session
		self.sess = tf.Session()

		# Append a 1 to fmaps to note that there is one layer in input layer
		fmaps = [1] + fmaps

		# Calculate number of layers
		self.numConvLayers = len(convsize) # Convolutional layers

		# Calculate post image size (after pooling)
		self.postImgSize = self.imgSize
		for i in range(self.numConvLayers):
			self.postImgSize = np.ceil(self.postImgSize/poolsize[i])

		# Define tf placeholders and variables
		self.x = tf.placeholder(tf.float32, shape=[None,imgSize,imgSize])
		self.y_ = tf.placeholder(tf.float32, shape=[None,numClass])

		# Store tf variables in a list, where each element is the variable for the corresponding layer
		self.W_conv = []
		self.b_conv = []
		self.h_conv = []
		self.h_pool = [] # The first element of this is the image layer

		# Put the image into h_pool
		self.h_pool.append(tf.reshape(self.x,[-1,imgSize,imgSize,1]))

		# Create all weight, bias and hidden convolutional layers
		for i in range(self.numConvLayers):
			self.W_conv.append(weight_variable([convsize[i], convsize[i], fmaps[i], fmaps[i+1]]))
			self.b_conv.append(bias_variable([fmaps[i+1]]))
			self.h_conv.append(tf.nn.relu(conv2d(self.h_pool[i], self.W_conv[i]) + self.b_conv[i]))
			self.h_pool.append(max_pool(self.h_conv[i], poolsize[i]))

		# Create fully connected layers
		# First fully connected layer is the flattened final convolutional layer
		self.flatFinalConvSize = int((self.postImgSize**2)*fmaps[-1]) # Flattened size of final convolutional layer
		self.W_fc1 = weight_variable([self.flatFinalConvSize,fcsize])
		self.b_fc1 = bias_variable([fcsize])

		self.h_pool_flat = tf.reshape(self.h_pool[-1], [-1, self.flatFinalConvSize])
		self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool_flat, self.W_fc1) + self.b_fc1)

		# Implement dropout
		# This is a technique used to reduce overfitting by forcing a random subset of neurons to
		# become inactive during training
		self.keep_prob = tf.placeholder(tf.float32)
		self.h_fc_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

		# Implement final softmax layer
		# Here we use softmax to output a probability distribution
		self.W_fc2 = weight_variable([fcsize, numClass])
		self.b_fc2 = bias_variable([numClass])

		# Output layer
		self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc_drop, self.W_fc2) + self.b_fc2)

		# Define cost function and training steps
		self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv), reduction_indices=[1]))
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

		# Define accuracy operations
		self.correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		self.sess.run(tf.initialize_all_variables())

		# Create saver to save and load variables
		self.saver = tf.train.Saver()


	## DATA INPUT AND TRAINING ####################################################

	# Function: inputData
	# Description: This function is used to input the data used to train, validate and test the network
	# Input Arguments:
	#	inputPKL: Path to a .pkl file which holds the training data in the following order:
	#		Image Data: An array of shape (N, imgSize, imgSize), where N is the number of data points
	#		Label Data: An array of shape (N, numClass)
	#	train_p: Percentage of data to use as training data
	#	valid_p: Percentage of data to use as validation data
	# Note: The rest of the data will be used as test data
	def inputData(self, inputPKL, train_p, valid_p):
		inputDataFile = open(inputPKL)
		self.images_full = np.array(pickle.load(inputDataFile))
		self.labels_full = np.array(pickle.load(inputDataFile))

		# Split data into training, validation, and test sets
		self.images_train, self.labels_train, self.images_valid, self.labels_valid, self.images_test, self.labels_test = splitData(self.images_full, self.labels_full, train_p, valid_p)

	# Function: train
	# Description: This function is used to train the network
	# Input Arguments:
	#	numStep: Number of training steps
	#	batchSize: Batch size of each step
	#	dispInterval: The interval to display progress
	def train(self, numStep, batchSize, dispInterval=100):
		for i in range(numStep):
			images_batch, labels_batch = nextBatch(self.images_train, self.labels_train, batchSize)

			# Print accuracy every 100th train_step
		  	if i%dispInterval == 0:
		  		train_accuracy = self.sess.run(self.accuracy, feed_dict={self.x:images_batch, self.y_: labels_batch, self.keep_prob: 1.0})
		  		print("step %d, training accuracy %g"%(i, train_accuracy))

		  	self.sess.run(self.train_step, feed_dict={self.x:images_batch, self.y_: labels_batch, self.keep_prob: 0.5})


	## SAVE AND LOAD FUNCTIONS ####################################################
	
	# Function: save
	# Description: Saves all TF variables into a checkpoint (.ckpt) file.
	# Input Arguments:
	#	savepath: Path to the store the output .ckpt file
	#	fname: Name of the .ckpt file
	def save(self, savepath, fname):
		full_savepath = savepath + fname + '.ckpt'
		output_path = self.saver.save(self.sess, full_savepath)
		print("Model saved in file: %s" % output_path)

	# Function: load
	# Description: Loads a previously saved checkpoint file.
	# NOTE: The CNNClassifier must be defined the EXACT same way as in the saved .ckpt file. This means that
	#	all input arguments in the constructor must be the same.
	# Input Arguments:
	# 	loadpath: Path to .ckpt file to load
	def load(self, loadpath):
		self.saver.restore(self.sess, loadpath)
		print("Model in file %s loaded successfully." % loadpath)


	## VISUALIZATION FUNCTIONS ####################################################

	# Function: plotImage
	# Description: Plots the specified stored image
	# Input Arguments:
	#	imgNum: The number of the image to plot
	def plotImage(self, imgNum):
		plt.imshow(self.images_full[imgNum,:,:], interpolation='nearest', cmap='Greys')
		plt.show()

	# Function: plotFeatMap_1
	# Description: Plots the weight matrix for the specified feature map in the first convolutional layer.
	# Input Arguments:
	#	featMapNum: The number of the feature map to plot
	def plotFeatMap_1(self, featMapNum):
		Wmap = self.sess.run(self.W_conv[0])
		plt.imshow(Wmap[:,:,0,featMapNum], interpolation='nearest', cmap='Greys')
		plt.show()