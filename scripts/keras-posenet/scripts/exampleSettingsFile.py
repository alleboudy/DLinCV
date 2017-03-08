import keras
import tensorflow as tf
BETA=5000	
pre='kc'
outputWeightspath  = '/usr/prakt/w065/trainedweights/2_3_'+pre+'.h5'
testweights='/usr/prakt/w065/trainedweights/2_3_'+pre+'.h5'
directory='/usr/prakt/w065/'+pre+'/'
startweight='/usr/prakt/w065/trainedweights/mergedweights.h5'#'../../mergedweights.h5'#'posenet.npy'
meanFile ='../../lstm-keras-tf/meanFiles/npy/'+pre+'.npy'
batchSize=100
logprefix=pre
traindata='dataset_train.txt'
testdata='dataset_test.txt'
optimizer =keras.optimizers.TFOptimizer(tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=0.1, use_locking=False, name='Adam'))
saveMean=False#is automatically set to true when training
 #(keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=0.00000001))
