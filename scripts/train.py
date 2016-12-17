import utilities
import posenet
import theano
import numpy as np



outputWeightspath = 'trainedweights.h5'
BETA = 250 #to 2000 for outdoor
directory = "/usr/prakt/w065/posenet/KingsCollege/"
dataset = 'dataset_train.txt'
 


def pose_loss(y_true, y_pred):
	print "####### IN THE POSE LOSS FUNCTION #####"
	return np.linalg.norm(y_true-y_pred) 

def rotation_loss(y_true, y_pred):
	print "####### IN THE ROTATION LOSS FUNCTION #####"
	return  BETA *np.linalg.norm(y_true-y_pred/np.linalg.norm(y_pred))







nb_epoch = 100
print "creating the model"
model =posenet.create_posenet('mergedweights.h5')
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='adam', loss=[pose_loss,rotation_loss])

for e in range(nb_epoch):
	print("epoch %d" % e)
	for X_batch, Y_batch in utilities.BatchGenerator(32,directory,dataset):
		#model.train(X_batch,Y_batch)
		#history = model.fit(X_batch, Y_batch,batch_size=32,shuffle=True,nb_epoch=1)
		history = model.fit(X_batch,{'cls3_fc_pose_wpqr': Y_batch[1], 'cls3_fc_pose_xyz': Y_batch[0]},
          nb_epoch=1, batch_size=32)
		print history.history['loss']

model.save_weights(outputWeightspath)
