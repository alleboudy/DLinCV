#for e in range(nb_epoch):
#    print("epoch %d" % e)
#    for X_train, Y_train in BatchGenerator():
#        model.fit(X_batch, Y_batch, batch_size=32, nb_epoch=1)
import numpy as np
import caffe
import cv2
directory = "/usr/prakt/w065/posenet/KingsCollege/"
dataset = 'dataset_train.txt'
meanFileLocation = 'imagemean.binaryproto'
def BatchGenerator(batchSize):
   
	print 'in batch generator'
        
        with open(directory+dataset) as f:
	    print 'opened file' 
            next(f)  # skip the 3 header lines
            next(f)
            next(f)
            while(True):
	        imagesBatch=[]
		posesBatch=[]
		for i in range(batchSize):
                    fname, p0,p1,p2,p3,p4,p5,p6 = next(f).split()
                    img = ResizeCropImage(cv2.imread(directory+fname )).astype(np.float32)
                    img = img.transpose((2, 0, 1))
                    img[0, :, :] -= meanImage[0,:,:].mean()
                    img[1, :, :] -= meanImage[1,:,:].mean()
                    img[2, :, :] -= meanImage[2,:,:].mean()
                    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
                    img = np.expand_dims(img, axis=0)
                    imagesBatch.append(img)
                    posesBatch.append((p0,p1,p2,p3,p4,p5,p6))
                print fname 

                yield imagesBatch,posesBatch


def getMean():
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( meanFileLocation, 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    return arr[0]


def ResizeCropImage(image):
    # we need to keep in mind aspect ratio so the image does
    # not look skewed or distorted -- therefore, we calculate
    # the ratio of the new image to the old image
    if image.shape[0]<image.shape[1]:
        r = 256.0 / image.shape[0]
        dim = ( 256,int(image.shape[1] * r))
    else:
        r = 256.0 / image.shape[1]
        dim = ( int(image.shape[0] * r),256)


    # perform the actual resizing of the image and show it
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)[0:224, 0:224]
    #cv2.imshow("resized", resized)
    #cv2.waitKey(0)
meanImage = getMean()
print ('mean')
for x in BatchGenerator(5):
	print len(x[0])

