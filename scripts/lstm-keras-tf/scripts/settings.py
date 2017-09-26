import keras
import tensorflow as tf
import datetime
ALPHA=1
BETA=1000
pre='heads'
shuffle=False
saveMean=False
#seqPrefix='seq2' # set to empty, should use all sequences present in a dataset for training
solDir='/usr/stud/alleboud/vision/'#'/work/alleboud/vision/'
datestring=datetime.datetime.now().strftime("%I-%M%p_%B-%d-%Y")
outputWeightspath  = solDir+'trainedweights/'+datestring+'_'+pre+'_LSTM.h5'#'KC_LSTM.h5'
oldmean=False#mean image = mean of means
testweights = solDir+'trainedweights/12-27PM_May-21-2017_'+pre+'_LSTM.h5'
directory=solDir+pre+'/'
startweight=solDir+'trainedweights/weights/posenet.npy'#mergedweights.h5'
#10-47AM_July-10-2017_sm_LSTM.h5'
#'../../mergedweights.h5'
#'/usr/prakt/w065/trainedweights/28_3_'+pre+'_LSTM.h5'#'/usr/prakt/w065/trainedweights/mergedweights.h5'#'../../mergedweights.h5'
#'lstmtfsmtrainedweights.h5'
#'batch30lstmtfsmtrainedweights.h5'#'../../mergedweights.h5'#'posenet.npy'
nb_epochs=100000
meanFile ='../meanFiles/npy/'+pre+'.npy'
batchSize=75
logprefix='LSTM_'+pre
stepSize = 3
traindata = solDir+'DLinCV/scripts/lstm-keras-tf/splittedOrderedSets/'+pre#+'orderedset_train.txt'
optimizer =keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#keras.optimizers.SGD(lr=0.0001, momentum=0.8, decay=0.0, nesterov=True)
#keras.optimizers.TFOptimizer(tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=0.00000001, use_locking=False, name='Adam'))
validationSplit=0
monitor ='loss'
testsetpath='../orderedSets/'+pre+'orderedset_test.txt'
distanceThreshold =8
angleThreshold = 8




historyloglocation ='{}{}traininghistory_{}.csv'.format(directory,logprefix,datestring)

 #(keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=0.00000001))
