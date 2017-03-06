import keras
import tensorflow as tf
ALPHA=1
BETA=500
pre='kc'
saveMean=False
outputWeightspath  = '/usr/prakt/w065/trainedweights/28_2_'+pre+'_LSTM.h5'#'KC_LSTM.h5'
oldmean=True
testweights = '/usr/prakt/w065/trainedweights/28_2_'+pre+'_LSTM.h5'
directory='/usr/prakt/w065/'+pre+'/'
startweight='/usr/prakt/w065/trainedweights/28_2_'+pre+'_LSTM.h5'#'/usr/prakt/w065/trainedweights/mergedweights.h5'#'../../mergedweights.h5'
#'lstmtfsmtrainedweights.h5'
#'batch30lstmtfsmtrainedweights.h5'#'../../mergedweights.h5'#'posenet.npy'
lr=0.0002
beta_1=0.9
beta_2=0.999
epsilon = 0.1
meanFile ='../meanFiles/npy/'+pre+'.npy'
batchSize=70
logprefix='LSTM_'+pre
stepSize = 3
traindata = '../orderedSets/'+pre+'orderedset_train.txt'
optimizer =keras.optimizers.TFOptimizer(tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=0.00000001, use_locking=False, name='Adam'))
#similarityThreshold=0.1
testsetpath='../orderedSets/'+pre+'orderedset_test.txt'
distanceThreshold =10
angleThreshold = 10
 #(keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=0.00000001))
