import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
from optparse import OptionParser

# PARAMETERS
imgSize = 44 # Size of one dimension of input image (assuming image is square)
convSize = 5 # Size of convolutional square
poolSize = 2 # Size of max pool
numTrainStep = 5000 # Number of training steps
batchSize = 50 # Size of batch

postImgSize = imgSize/poolSize/poolSize # Size of image after 2 layers of pooling

# Options Parsing
parser = OptionParser()
parser.add_option("-i", "--input", action="store", type="string", dest="inputFile", help="path to input pkl file", metavar="INPUTPKL")

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
	perm = np.random.permutation(numData)
	featList = featList[perm][:][:]
	labelList1 = labelList1[perm][:]

	# Number of data points in each data set
	train_num = int(round(numData*train_p))
	valid_num = int(round(numData*valid_p))

	# Split up data sets
	feat_train = featList[0:train_num][:][:]
	label_train = labelList1[0:train_num][:]

	feat_valid = featList[train_num:train_num+valid_num][:][:]
	label_valid = labelList1[train_num:train_num+valid_num][:]

	feat_test = featList[train_num+valid_num:][:][:]
	label_test = labelList1[train_num+valid_num:][:]

	return feat_train, label_train, feat_valid, label_valid, feat_test, label_test

# Function to create a random batch of data sets
def nextBatch(featList, labelList1, batchsize):
	numData = len(labelList1)

	# Randomly shuffle data
	perm = np.random.permutation(numData)
	featList = featList[perm][:][:]
	labelList1 = labelList1[perm][:]

	feat_batch = featList[:batchsize][:][:]
	label_batch = labelList1[:batchsize][:]

	return feat_batch, label_batch

# Functions to initialize weight and bias
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

# Convolution and pooling functions
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x, n=2):
	return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')

# Import cell data
cellinput = open(inputFile)
cellImg = np.array(pickle.load(cellinput))
cellLabel = np.array(pickle.load(cellinput))

# Open session
sess = tf.Session()

# Split data
cell_train, label_train, cell_valid, label_valid, cell_test, label_test = splitData(cellImg, cellLabel, 0.75, 0)	

# Define placeholder variables
x = tf.placeholder(tf.float32, shape=[None, imgSize, imgSize])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

# First convolutional layer
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,imgSize,imgSize,1])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1, poolSize)

# Second convolutional layer
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2, poolSize)

# Densely connected layer
W_fc1 = weight_variable([(postImgSize**2)*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, (postImgSize**2)*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Implement dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Implement softmax
W_fc2 = weight_variable([1024,2])
b_fc2 = bias_variable([2])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Train network
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

# Create saver
saver = tf.train.Saver()

for i in range(numTrainStep):
	cell_batch, label_batch = nextBatch(cell_train, label_train, batchSize)

	# Print accuracy every 100th train_step
  	if i%100 == 0:
  		train_accuracy = sess.run(accuracy, feed_dict={x:cell_batch, y_: label_batch, keep_prob: 1.0})
  		print("step %d, training accuracy %g"%(i, train_accuracy))

  	sess.run(train_step, feed_dict={x:cell_batch, y_: label_batch, keep_prob: 0.5})

train_accuracy = sess.run(accuracy, feed_dict={x: cell_test, y_: label_test, keep_prob: 1.0})
print("test accuracy %g"%train_accuracy)

save_path = saver.save(sess, "cellconv_model.ckpt")
print("Model saved in file: %s" % save_path)


