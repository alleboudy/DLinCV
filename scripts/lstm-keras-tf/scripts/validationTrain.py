import utilitiesTrain as utilities
#import posenet
import old_cnn_lstm as cnn_lstm
import theano
import numpy as np
import keras
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras import backend as K
import time
import settings
from keras.callbacks import ModelCheckpoint
import datetime
from validate import validate
outputWeightspath =settings.outputWeightspath #'oldhospitaltrainedweights.h5'
BETA = settings.BETA #to 2000 for outdoor
ALPHA=settings.ALPHA
directory = settings.directory#"/usr/prakt/w065/posenet/OldHospital/"
settings.saveMean=True
#dataset = 'dataset_train.txt'
#historyloglocation = '{}traininghistory_{}.txt'.format(directory,str(time.time()))
historyloglocation =settings.historyloglocation #'{}{}traininghistory_{}.csv'.format(directory,settings.logprefix,datetime.datetime.now().strftime("%I-%M%p_%B-%d-%Y"))
with open(historyloglocation,"a+") as f:
	f.write('validationSplit:{}\ncorrespondingWeights:{}\nbatchSize:{}\nBeta:{}\nmeanOfMeansImage:{}\nstartingweights:{}\n'.format(str(settings.validationSplit), str(outputWeightspath),str(settings.batchSize),str(settings.BETA),str(settings.oldmean),settings.startweight))
#Validationhistoryloglocation = '{}validationhistory_{}.txt'.format(directory,str(time.time()))
#startweight = 'oldhospitaltrainedweights.h5' 
startweight= settings.startweight #'../mergedweights.h5'

#def pose_loss12(y_true, y_pred):
#	print "####### IN THE POSE LOSS FUNCTION #####"
#	return 0.3* K.sqrt(K.sum(K.square((y_pred - y_true)))) 

#def rotation_loss12(y_true, y_pred):
#	print "####### IN THE ROTATION LOSS FUNCTION #####"
#	return 150* K.sqrt(K.sum(K.square(y_true-y_pred)))


def pose_loss3(y_true, y_pred):
        print "####### IN THE POSE LOSS FUNCTION #####"
        return ALPHA* K.sqrt(K.sum(K.square((y_pred - y_true))))

def rotation_loss3(y_true, y_pred):
        print "####### IN THE ROTATION LOSS FUNCTION #####"
        return  BETA *K.sqrt(K.sum(K.square((y_true-y_pred))))

print "creating the model"
model =cnn_lstm.create_cnn_lstm(startweight)


class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		with open(historyloglocation,"a+") as f:
			f.write('{},{}\n'.format('val_loss', 'loss'))
			

	def on_batch_end(self, batch, logs={}):
		#print (self.params)
		pass

	def on_epoch_end(self, epoch, logs={}):
		valTr,valRo,_ = validate(model) 
		with open(historyloglocation,"a+") as f:
			f.write('{},{},{},{}\n'.format(logs.get('val_loss'), logs.get('loss'),valTr,valRo))
#batchSize=25
nb_epochs = settings.nb_epochs
#print "creating the model"
#model =cnn_lstm.create_cnn_lstm(startweight)
sgd = settings.optimizer
#sgd = SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss=[pose_loss3,rotation_loss3])

#for e in range(nb_epoch):
#print("epoch %d" % e)
datasource = utilities.get_data(settings.traindata)

#data_gen = utilities.gen_data_batch(datasource)


allX = []
allY =[]
allZ=[]
YZ=[]
for X,Y,Z in utilities.get_data_examples(datasource):
	allX.append(X)
	allY.append(Y)
	allZ.append(Z)

with open(historyloglocation,"a+") as f:
     f.write('total number of samples:{}\n'.format(len(allY)))


YZ.append(np.asarray(allY))
YZ.append(np.asarray(allZ))
checkpointer = ModelCheckpoint(filepath=outputWeightspath, verbose=1, save_best_only=True,monitor='val_loss')
if settings.validationSplit==0:
	checkpointer = ModelCheckpoint(filepath=outputWeightspath, verbose=1, save_best_only=True,monitor='loss')
history = LossHistory()
model.fit(np.asarray(allX),YZ,
	  epochs=nb_epochs,batch_size=settings.batchSize,validation_split=settings.validationSplit,callbacks=[history,checkpointer])

#for i in range(nb_epochs):

#	X_batch, Y_batch = next(data_gen)
		#model.train(X_batch,Y_batch)
		#history = model.fit(X_batch, Y_batch,batch_size=32,shuffle=True,nb_epoch=1)
	#print Y_batch[0].shape
	#print Y_batch[1].shape
	#print len(Y_batch)	
#	history = model.fit(X_batch,Y_batch,
 #         nb_epoch=1,batch_size=utilities.batchSize)
	#history = model.fit(X_batch,{'pose_wpqr': Y_batch[1], 'pose_xyz': Y_batch[0]},
     #     nb_epoch=1,batch_size=utilities.batchSize)
#	print settings.logprefix
#	print 'epoch: ', i
#	print 'loss: ',history.history['loss'][0]
#	with open(historyloglocation,"a+") as f:
 #               f.write('{},{}\n'.format(str(i), str(history.history['loss'][0])))

#	if i%25==0:
#		print 'saving trained weights in'
#		print outputWeightspath
#		model.save_weights(outputWeightspath)



#model.save_weights(outputWeightspath)
