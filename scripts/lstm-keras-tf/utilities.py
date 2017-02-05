import numpy as np
#import caffe
import cv2
import random
import settings
directory = settings.directory  
# "/usr/prakt/w065/posenet/OldHospital/"
dataset = 'dataset_train.txt'
meanFile = settings.meanFile  # 'oldhospitaltrainmean.binaryproto'
batchSize = settings.batchSize
# resizes a given image so that the smallest dimension is 256 then crops
# 244X244 from the middle of it


def ResizeCropImage(image):
    # we need to keep in mind aspect ratio so the image does
    # not look skewed or distorted -- therefore, we calculate
    # the ratio of the new image to the old image
    if image.shape[0] < image.shape[1]:
        r = 256.0 / image.shape[0]
        dim = (int(image.shape[1] * r), 256)
    else:
        r = 256.0 / image.shape[1]
        dim = (256, int(image.shape[0] * r))
        #dim is (cols,rows)

    # perform the actual resizing of the image and show it
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    centerWidth = image.shape[1] / 2
    centerHeight = image.shape[0] / 2
    return image[centerHeight - 112:centerHeight + 112, centerWidth - 112:centerWidth + 112]
    #cv2.imshow("resized", resized)
    # cv2.waitKey(0)

# extracts the mean image form a given mean file


def getMean():
    #blob = caffe.proto.caffe_pb2.BlobProto()
    #data = open( meanFileLocation, 'rb' ).read()
    # blob.ParseFromString(data)
    #arr = np.array( caffe.io.blobproto_to_array(blob) )
    return np.load(meanFile)
    # return #arr[0]

# outputs two lists of numpy arrays
meanImage = getMean()


def get_data():
    imagesBatch = []
    po1 = []
    po2 = []
    # print 'in batch generator'
    # while(True):
   # while(True):
   # print 'lol1'
    # while(True):
    # print'lol2'
    with open(directory + dataset) as f:
        print 'opened file'
        next(f)  # skip the 3 header lines
        next(f)
        next(f)
   # while(True):
        # print 'lol3'

        for line in f:
            fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
            img = ResizeCropImage(cv2.imread(
                directory + fname)).astype(np.float32)
            img = img.transpose((2, 0, 1))
            img[0, :, :] -= meanImage[0, :, :].mean()
            img[1, :, :] -= meanImage[1, :, :].mean()
            img[2, :, :] -= meanImage[2, :, :].mean()
            img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
#                    img = np.expand_dims(img, axis=0)
            imagesBatch.append(img)
            po1.append(np.array((p0, p1, p2)))
            po2.append(np.array((p3, p4, p5, p6)))


#		print po1.shape,p2.shape
    return (np.asarray(imagesBatch), [np.asarray(po1), np.asarray(po2)])


def gen_data(source):
    while True:
        indices = range(len(source[0]))
        random.shuffle(indices)
        for i in indices:
            image = source[0][i]
            pose_x = source[1][0][i]
            pose_q = source[1][1][i]
            yield image, pose_x, pose_q


def gen_data_batch(source):
    data_gen = gen_data(source)
    while True:
        image_batch = []
        pose_x_batch = []
        pose_q_batch = []
        for _ in range(batchSize):
            image, pose_x, pose_q = next(data_gen)
            image_batch.append(image)
            pose_x_batch.append(pose_x)
            pose_q_batch.append(pose_q)
        yield np.array(image_batch), [np.array(pose_x_batch), np.array(pose_q_batch)]
