import numpy as np
import cnn_lstm
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
directory = settings.directory  # "/usr/prakt/w065/posenet/sm/"

dataset = 'dataset_test.txt'
#outputDirectory = "/usr/prakt/w065/posenet/TFData/"
#meanFileLocation = 'smmean.binaryproto'
# 'tfsmtrainedweights.h5'#'75batbhessmtrainedweights.h5'#'smtrainedweights.h5'
weightsfile = settings.outputWeightspath
# weightsfile='shoptrainedweights.h5'
poses = []  # will contain poses followed by qs
images = []

# limitingCounter=3
# def getMean():
#blob = caffe.proto.caffe_pb2.BlobProto()
#data = open( meanFileLocation, 'rb' ).read()
# blob.ParseFromString(data)
#arr = np.array( caffe.io.blobproto_to_array(blob) )
# return arr[0]


# def ResizeCropImage(image):
# we need to keep in mind aspect ratio so the image does
# not look skewed or distorted -- therefore, we calculate
# the ratio of the new image to the old image
#   if image.shape[0]<image.shape[1]:
#      r = 256.0 / image.shape[0]
#     dim = ( 256,int(image.shape[1] * r))
# else:
#   r = 256.0 / image.shape[1]
#  dim = ( int(image.shape[0] * r),256)

# perform the actual resizing of the image and show it
# return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)[0:224, 0:224]
#cv2.imshow("resized", resized)
# cv2.waitKey(0)

meanImage = utilities.getMean()
# print meanImage.shape
# Test pretrained model
model = cnn_lstm.create_cnn_lstm(weightsfile)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
#meantrasnformed = meanImage
#meantrasnformed[:,:,[0,1,2]]  = meanImage[:,:,[2,1,0]]
#meantrasnformed =  np.expand_dims(meantrasnformed, axis=0)
posxs = []
posqs = []
imgs = []
seq = [
    'seq13/frame00155.png 27.616028 -0.242176 54.205308 0.362095 -0.008852 0.924934 -0.115366',
    'seq13/frame00156.png 26.839205 -0.252827 54.305833 0.377098 -0.008740 0.916330 -0.134392',
    'seq13/frame00157.png 26.084117 -0.210158 54.388376 0.355471 -0.009817 0.925107 -0.133134'

]
counter = 0
with open(settings.testsetpath) as f:
    inputs = np.zeros([1, 3, 3, 224, 224])
    labels = np.zeros([1, 3, 7])
    for line in f:
        fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
        p0 = float(p0)
        p1 = float(p1)
        p2 = float(p2)
        p3 = float(p3)
        p4 = float(p4)
        p5 = float(p5)
        p6 = float(p6)
        img = utilities.ResizeCropImage(
            imread(directory + fname)).astype(np.float32)

        img = img.transpose((2, 0, 1))
        img[0, :, :] -= meanImage[0, :, :].mean()
        img[1, :, :] -= meanImage[1, :, :].mean()
        img[2, :, :] -= meanImage[2, :, :].mean()
        inputs[0, counter % settings.stepSize, :, :, :] = img
	labels[0, counter % settings.stepSize, :] = np.asarray(p0,p1,p2,p3,p4,p5,p6)
        counter += 1
        if counter % 3 == 0:
            out = model.predict(inputs)
	   # print out
            posx = out[0]
            posq = out[1]
            print "actual:"
            actualx = (p0, p1, p2)
            actualq = (p3, p4, p5, p6)
            q1 = actualq / np.linalg.norm(actualq)
            q2 = posq / np.linalg.norm(posq)
            d = abs(np.sum(np.multiply(q1, q2)))
            theta = 2 * np.arccos(d) * 180 / np.pi
            errx = np.linalg.norm(actualx - posx)
            posxs.append(errx)
            posqs.append(theta)
            print 'errx ', errx, ' m and ', 'errq ', theta, ' degrees'
            inputs = np.zeros([1, 3, 3, 224, 224])
	    labels = np.zeros([1, 3, 7])
print 'median error', np.median(posxs), ' m and ', np.median(posqs), ' degrees'

    # poses.append((p0,p1,p2,p3,p4,p5,p6))
    # images.append(directory+fname)

    # imgs.append(img)
# print len(imgs)
# print 	np.asarray(imgs).shape
  # note: the model has three outputs
# out = model.predict(np.expand_dims(np.asarray(imgs),axis=0))  # note: the model has three outputs
# for i in range(len(out)):
#	for j in range(len(out[i])):
#		out[i][j]+=meanout[i][j]
# print np.argmax(out[2])
# print out
# print "predcited:"
