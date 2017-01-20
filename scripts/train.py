import utilities
import posenet
import theano
import numpy as np
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras import backend as K
import time
outputWeightspath = 'oldhospitaltrainedweights.h5'
BETA = 500 #to 2000 for outdoor
directory = "/usr/prakt/w065/posenet/OldHospital/"
#dataset = 'dataset_train.txt'
historyloglocation = '{}OldHospitaltraininghistory_{}.txt'.format(directory,str(time.time()))

Validationhistoryloglocation = '{}validationhistory_{}.txt'.format(directory,str(time.time()))
startweight = 'oldhospitaltrainedweights.h5' 


def pose_loss12(y_true, y_pred):
	print "####### IN THE POSE LOSS FUNCTION #####"
	return 0.3* K.sqrt(K.sum(K.square((y_pred - y_true)))) 

def rotation_loss12(y_true, y_pred):
	print "####### IN THE ROTATION LOSS FUNCTION #####"
	return 150* K.sqrt(K.sum(K.square(y_true-y_pred)))


def pose_loss3(y_true, y_pred):
        print "####### IN THE POSE LOSS FUNCTION #####"
        return K.sqrt(K.sum(K.square((y_pred - y_true))))

def rotation_loss3(y_true, y_pred):
        print "####### IN THE ROTATION LOSS FUNCTION #####"
        return  BETA *K.sqrt(K.sum(K.square((y_true-y_pred))))




#batchSize=25
nb_epochs = 30000
print "creating the model"
model =posenet.create_posenet(startweight)
sgd = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=0.00000001)
#sgd = SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss=[pose_loss12,rotation_loss12,pose_loss12,rotation_loss12,pose_loss3,rotation_loss3])

#for e in range(nb_epoch):
#print("epoch %d" % e)
datasource = utilities.get_data()

data_gen = utilities.gen_data_batch(datasource)
for i in range(nb_epochs):

	X_batch, Y_batch = next(data_gen)
		#model.train(X_batch,Y_batch)
		#history = model.fit(X_batch, Y_batch,batch_size=32,shuffle=True,nb_epoch=1)
	
	history = model.fit(X_batch,{'cls1_fc_pose_wpqr': Y_batch[1], 'cls1_fc_pose_xyz': Y_batch[0],'cls2_fc_pose_wpqr': Y_batch[1], 'cls2_fc_pose_xyz': Y_batch[0],'cls3_fc_pose_wpqr': Y_batch[1], 'cls3_fc_pose_xyz': Y_batch[0]},
          nb_epoch=1,batch_size=utilities.batchSize)
	print 'epoch: ', i
	print 'loss: ',history.history['loss'][0]
	with open(historyloglocation,"a+") as f:
                f.write('{},{}\n'.format(str(i), str(history.history['loss'][0])))

	if i%25==0:
		print 'saved trained weights'
		model.save_weights(outputWeightspath)


model.save_weights(outputWeightspath)
