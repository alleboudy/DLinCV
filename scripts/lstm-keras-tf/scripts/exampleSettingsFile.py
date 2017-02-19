import keras
import tensorflow as tf
outputWeightspath  = '/usr/prakt/w065/trainedweights/SM_LSTM.h5'#'KC_LSTM.h5'
testweights = '/usr/prakt/w065/trainedweights/SM_LSTM_SEQ.h5'
BETA = 500
directory='/usr/prakt/w065/posenet/sm/'
startweight='../../mergedweights.h5'
#'lstmtfsmtrainedweights.h5'
#'batch30lstmtfsmtrainedweights.h5'#'../../mergedweights.h5'#'posenet.npy'
lr=0.0002
beta_1=0.9
beta_2=0.999
epsilon = 0.1
meanFile ='../meanFiles/npy/smmean.npy'
batchSize=30
logprefix='LSTM_KC'
stepSize = 3
traindata = '../orderedSets/SMorderedtrain.txt'
optimizer =keras.optimizers.TFOptimizer(tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=0.00000001, use_locking=False, name='Adam'))
#similarityThreshold=0.1
testsetpath='../orderedSets/SMorderedtest.txt'
distanceThreshold =10
angleThreshold = 10
 #(keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=0.00000001))
