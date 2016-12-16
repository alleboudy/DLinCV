import numpy as np
import caffe
import cv2

directory = "/usr/prakt/w065/posenet/KingsCollege/"
dataset = 'dataset_train.txt'
dataLocation= 'directory+dataset'


#resizes a given image so that the smallest dimension is 256 then crops 244X244 from the middle of it
def ResizeCropImage(image):
    # we need to keep in mind aspect ratio so the image does
    # not look skewed or distorted -- therefore, we calculate
    # the ratio of the new image to the old image
    if image.shape[0]<image.shape[1]:
        r = 256.0 / image.shape[0]
        dim = ( int(image.shape[1] * r),256)
    else:
        r = 256.0 / image.shape[1]
        dim = (256, int(image.shape[0] * r))
        #dim is (cols,rows) 

    # perform the actual resizing of the image and show it
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    centerWidth = image.shape[1]/2
    centerHeight = image.shape[0]/2
    return image[centerHeight-112:centerHeight+112, centerWidth-112:centerWidth+112]
    #cv2.imshow("resized", resized)
    #cv2.waitKey(0)

#extracts the mean image form a given mean file
def getMean(meanFileLocation = 'imagemean.binaryproto'):
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( meanFileLocation, 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    return arr[0]

#outputs two lists of numpy arrays
meanImage =getMean()
def BatchGenerator(batchSize,directory,dataset):
   
	#print 'in batch generator'
        
        with open(directory+dataset) as f:
	    #print 'opened file' 
            next(f)  # skip the 3 header lines
            next(f)
            next(f)
            while(True):
	        imagesBatch=[]
	        po1=[]
		po2=[]
		for i in range(batchSize):
                    fname, p0,p1,p2,p3,p4,p5,p6 = next(f).split()
                    img = ResizeCropImage(cv2.imread(directory+fname )).astype(np.float32)
                    img = img.transpose((2, 0, 1))
                    img[0, :, :] -= meanImage[0,:,:].mean()
                    img[1, :, :] -= meanImage[1,:,:].mean()
                    img[2, :, :] -= meanImage[2,:,:].mean()
                    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
#                    img = np.expand_dims(img, axis=0)
                    imagesBatch.append(img)
		    po1.append(np.array((p0,p1,p2)))
		    po2.append(np.array((p3,p4,p5,p6)))
                  
                #print fname 
#		print po1.shape,p2.shape
                yield np.asarray(imagesBatch),[np.asarray(po1),np.asarray(po2)]
