import numpy as np
import old_cnn_lstm as cnn_lstm
from scipy.misc import imread, imresize
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from custom_layers import PoolHelper  # ,LRN
#import caffe
import cv2
import utilities
from LRN2D import LRN2D as LRN
import settings
from similarityMeasures import getError
#directory = settings.directory 
#datasource = utilities.get_data(settings.testsetpath)
#datagen = utilities.limited_gen_data(datasource)
#settings.saveMean=False
#weightsfile = settings.testweights

def validate(model):
	directory = settings.directory
	datasource = utilities.get_data(settings.testsetpath)
	datagen = utilities.limited_gen_data(datasource)
	settings.saveMean=False

	#model = cnn_lstm.create_cnn_lstm(weightsfile)
	#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	#model.compile(optimizer=sgd, loss='categorical_crossentropy')
	print('\nin validate method!!!!!!')
	print(type(model))
	posxs = []
	posqs = []
	howmanyaccepted=0
	counter = 0
	print('looping on test set!')
	for ims,xs,qs in datagen:
	    print(len(ims))
	    howmanyaccepted+=1
	    print howmanyaccepted
	    inputs = np.zeros([1, 3, 3, 224, 224])
	    inputs[0,:]=ims
	    out = model.predict(inputs)
	    posx = out[0][0][1]#.mean(0)#xyz
	    posq = out[1][0][1]#.mean(0)#wpqr
	    actualx = xs[1]#.mean(0)
	    actualq = qs[1]#.mean(0)
	    errx, theta = getError(posx,posq,actualx,actualq)
	    posxs.append(errx)
	    posqs.append(theta)
	    print('error should report here!')
	    print 'errx ', errx, ' m and ', 'errq ', theta, ' degrees'
	return  np.median(posxs),  np.median(posqs),howmanyaccepted
