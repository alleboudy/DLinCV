import numpy as np
import posenet
from scipy.misc import imread, imresize
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from custom_layers import PoolHelper#,LRN
import caffe
import cv2
import utilities
from LRN2D import LRN2D as LRN
directory = "/usr/prakt/w065/kc/"

dataset = 'dataset_test.txt'
outputDirectory = "/usr/prakt/w065/posenet/TFData/"
meanFileLocation = 'smmean.binaryproto'
weightsfile='kc.h5'#'75batbhessmtrainedweights.h5'#'smtrainedweights.h5'
#weightsfile='shoptrainedweights.h5'
poses = [] #will contain poses followed by qs
images = []

#limitingCounter=3
def getMean():
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( meanFileLocation, 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    return arr[0]



#def ResizeCropImage(image):
    # we need to keep in mind aspect ratio so the image does
    # not look skewed or distorted -- therefore, we calculate
    # the ratio of the new image to the old image
 #   if image.shape[0]<image.shape[1]:
  #      r = 256.0 / image.shape[0]
   #     dim = ( 256,int(image.shape[1] * r))
    #else:
     #   r = 256.0 / image.shape[1]
      #  dim = ( int(image.shape[0] * r),256)


    # perform the actual resizing of the image and show it
    #return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)[0:224, 0:224]
    #cv2.imshow("resized", resized)
    #cv2.waitKey(0)

meanImage = getMean()
#print meanImage.shape 
         # Test pretrained model
model = posenet.create_posenet(weightsfile)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
#meantrasnformed = meanImage
#meantrasnformed[:,:,[0,1,2]]  = meanImage[:,:,[2,1,0]]
#meantrasnformed =  np.expand_dims(meantrasnformed, axis=0)
posxs=[]
posqs=[]
#meanout = model.predict(meantrasnformed)
with open(directory+dataset) as f:
    next(f)  # skip the 3 header lines
    next(f)
    next(f)
    for line in f:
        #if limitingCounter ==0:
        #   break
	#limitingCounter-=1
        fname, p0,p1,p2,p3,p4,p5,p6 = line.split()
        p0 = float(p0)
        p1 = float(p1)
        p2 = float(p2)
        p3 = float(p3)
        p4 = float(p4)
        p5 = float(p5)
        p6 = float(p6)
        #poses.append((p0,p1,p2,p3,p4,p5,p6))
        #images.append(directory+fname)
        img = utilities.ResizeCropImage(imread(directory+fname )).astype(np.float32)

        img = img.transpose((2, 0, 1))
        img[0,:, :] -= meanImage[0,:,:].mean()
        img[1,:, :] -= meanImage[1,:,:].mean()
        img[2,:, :] -= meanImage[2,:,:].mean()
        #img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
        img = np.expand_dims(img, axis=0)
        out = model.predict(img) # note: the model has three outputs
	#for i in range(len(out)):
	#	for j in range(len(out[i])):
	#		out[i][j]+=meanout[i][j]
            #print np.argmax(out[2])
        
	print "predcited:"
        posx= out[2]
	posq= out[5]
        print "actual:"
        actualx= (p0,p1,p2)
	actualq=(p3,p4,p5,p6)
	q1= actualq/np.linalg.norm(actualq)
	q2 = posq/np.linalg.norm(posq)
	d = abs(np.sum(np.multiply(q1,q2)))
	theta =2*np.arccos(d)*180/np.pi
	errx=np.linalg.norm(actualx-posx)
	posxs.append(errx)
	posqs.append(theta)
	print 'errx ', errx, ' m and ', 'errq ', theta,' degrees'
print 'median error',np.median(posxs), ' m and ', np.median(posqs), ' degrees'

