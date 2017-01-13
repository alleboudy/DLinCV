import utilities
import posenet
import theano
import numpy as np
from keras.optimizers import SGD
from keras import backend as K

outputWeightspath = 'trainedweights.h5'
BETA = 500 #to 2000 for outdoor
directory = "/usr/prakt/w065/posenet/OldHospital/"
dataset = 'dataset_train.txt'
 


def pose_loss12(y_true, y_pred):
	print "####### IN THE POSE LOSS FUNCTION #####"
	return 0.3* K.abs(y_pred - y_true) 

def rotation_loss12(y_true, y_pred):
	print "####### IN THE ROTATION LOSS FUNCTION #####"
	return  150* K.abs(y_true-y_pred/K.abs(y_pred))


def pose_loss3(y_true, y_pred):
        print "####### IN THE POSE LOSS FUNCTION #####"
        return K.abs(y_pred - y_true)

def rotation_loss3(y_true, y_pred):
        print "####### IN THE ROTATION LOSS FUNCTION #####"
        return  BETA *K.abs(y_true-y_pred/K.abs(y_pred))





nb_epoch = 30000
print "creating the model"
model =posenet.create_posenet('mergedweights.h5')
sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss=[pose_loss12,rotation_loss12,pose_loss12,rotation_loss12,pose_loss3,rotation_loss3])

for e in range(nb_epoch):
	print("epoch %d" % e)
	for X_batch, Y_batch in utilities.BatchGenerator(32,directory,dataset):
		#model.train(X_batch,Y_batch)
		#history = model.fit(X_batch, Y_batch,batch_size=32,shuffle=True,nb_epoch=1)
		history = model.fit(X_batch,{'cls1_fc_pose_wpqr': Y_batch[1], 'cls1_fc_pose_xyz': Y_batch[0],'cls2_fc_pose_wpqr': Y_batch[1], 'cls2_fc_pose_xyz': Y_batch[0],'cls3_fc_pose_wpqr': Y_batch[1], 'cls3_fc_pose_xyz': Y_batch[0]},
          nb_epoch=1, batch_size=32)
		print history.history['loss']
		if e%500==0:
			model.save_weights(outputWeightspath)


model.save_weights(outputWeightspath)
