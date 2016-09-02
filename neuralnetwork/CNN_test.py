#
# Last modified: 1 September 2016
# Authors: Darrick Lee <y.l.darrick@gmail.com>
# Description: This is a test script to show how the CNNClassifier works.

from CNNClassifier import CNNClassifier

## DEFINING PARAMETERS ########################################################

# Determine whether or not you want to load data
loaddata = 1 # (0: no, 1:yes)
load_file = 'sample_cnn.ckpt'

# Define parameters for the CNN
imgSize = 44 # Size of one dimension of the input images
numClass = 2 # Number of output classes (in this case 2: Red and Green)
fmaps = [32, 64] # Number of feature maps in each convolution layer (in this case, 32 in first layer and 64 in second layer)
convSize = [5, 5] # Size of square convolutional filter in each layer (in this case, 5x5 for both convolutional layers)
poolSize = [2, 2] # Size of square pooling filter in each layer (in this case, 2x2 for both layers)
fcsize = 1024 # Number of neurons in the fully connected layer at the end (currently, we only have one fully connected layer, which is connected to the softmax output)

# Define parameters for the data
pkl_file = 'labelled_data.pkl' # PKL file where data is stored (first variable is the image data, and second variable is the label data)
train_p = 0.85 # Percentage of data to use as training data
valid_p = 0 # Percentage of data to use as validation data
# Note: The remaining data will be used as test data

# Define training parameters
trainSteps = 200 # Number of steps to train
batchSize = 100 # Size of each batch

# Define save parameters
savePath = '' # Define path to output folder (in this case, just save in current directory)
saveName = 'sample_cnn' # Saved file name (result will be a .ckpt file)

## USING CNNCLASSIFIER ########################################################

# Define the CNNClassifier
CNN = CNNClassifier(imgSize, numClass, fmaps, convSize, poolSize, fcsize)

# Input data into CNNClassifier
CNN.inputData(pkl_file, train_p, valid_p)

# Load data if required
if(loaddata == 1):
	CNN.load(load_file)

# Train classifier
CNN.train(trainSteps, batchSize)

# Save current state
CNN.save(savePath, saveName)

