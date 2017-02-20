import numpy as np
#import caffe
import cv2
import random
import settings
from similarityMeasures import getError
directory = settings.directory
# "/usr/prakt/w065/posenet/OldHospital/"
data = settings.traindata  # 'KCorderedtrain.txt'
print data
meanFile = settings.meanFile  # 'oldhospitaltrainmean.binaryproto'
batchSize = settings.batchSize
# resizes a given image so that the smallest dimension is 256 then crops
# 244X244 from the middle of it
# samplesCounter=0


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


def get_data(dataset=data):
    imagesBatch = []
    po1 = []
    po2 = []
    # print 'in batch generator'
    # while(True):
   # while(True):
   # print 'lol1'
    # while(True):
    # print'lol2'
    print dataset
    with open( dataset) as f:
        print 'reading training data'
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
            #img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
#                    img = np.expand_dims(img, axis=0)
            imagesBatch.append(img)
            po1.append(np.array((np.float(p0), np.float(p1), np.float(p2))))
            po2.append(
                np.array((np.float(p3), np.float(p4), np.float(p5), np.float(p6))))


#		print po1.shape,p2.shape
    return (np.asarray(imagesBatch), [np.asarray(po1), np.asarray(po2)])


def limited_gen_data(source):
    # while True:
    indices = range(len(source[0]))
    #random.shuffle(indices)
    for i in indices:
        image = source[0][i]
        image_left = source[0][max(0, i - 1)]
        image_right = source[0][min(i + 1, len(source[0]) - 1)]
        pose_x = source[1][0][i]
        pose_q = source[1][1][i]
        pose_x_left = source[1][0][max(0, i - 1)]
        pose_q_left = source[1][1][max(0, i - 1)]
        pose_x_right = source[1][0][min(i + 1, len(source[0]) - 1)]
        pose_q_right = source[1][1][min(i + 1, len(source[0]) - 1)]
       # print type(pose_x)
       # print pose_x
        # print pose_q
        # print pose_x_left
        m1_2, a1_2 = getError(pose_x, pose_q, pose_x_left, pose_q_left)
        m2_3, a2_3 = getError(pose_x, pose_q, pose_x_right, pose_q_right)
        m1_3, a1_3 = getError(pose_x_left, pose_q_left,
                              pose_x_right, pose_q_right)

        if m1_2 > settings.distanceThreshold or m2_3 > settings.distanceThreshold:
            continue
        if a1_2 > settings.angleThreshold or a2_3 > settings.angleThreshold:
            continue
        #global samplesCounter
        # samplesCounter+=1
        # print samplesCounter
        yield np.asarray([image_left, image, image_right]), np.asarray([pose_x, pose_x_left, pose_x_right]), np.asarray([pose_q, pose_q_left, pose_q_right])
        #, pose_x_right,pose_q_right


def gen_data(source):
    while True:
        indices = range(len(source[0]))
        random.shuffle(indices)
        for i in indices:
            image = source[0][i]
            image_left = source[0][max(0, i - 1)]
            image_right = source[0][min(i + 1, len(source[0]) - 1)]
            pose_x = source[1][0][i]
            pose_q = source[1][1][i]
            pose_x_left = source[1][0][max(0, i - 1)]
            pose_q_left = source[1][1][max(0, i - 1)]
            pose_x_right = source[1][0][min(i + 1, len(source[0]) - 1)]
            pose_q_right = source[1][1][min(i + 1, len(source[0]) - 1)]
       # print type(pose_x)
       # print pose_x
            # print pose_q
            # print pose_x_left
            m1_2, a1_2 = getError(pose_x, pose_q, pose_x_left, pose_q_left)
            m2_3, a2_3 = getError(pose_x, pose_q, pose_x_right, pose_q_right)
            m1_3, a1_3 = getError(pose_x_left, pose_q_left,
                                  pose_x_right, pose_q_right)

            if m1_2 > settings.distanceThreshold or m2_3 > settings.distanceThreshold:
                continue
            if a1_2 > settings.angleThreshold or a2_3 > settings.angleThreshold:
                continue
            #global samplesCounter
            # samplesCounter+=1
            # print samplesCounter
            yield np.asarray([image_left, image, image_right]), np.asarray([pose_x, pose_x_left, pose_x_right]), np.asarray([pose_q, pose_q_left, pose_q_right])
            #, pose_x_right,pose_q_right


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
        yield np.asarray(image_batch), [np.asarray(pose_x_batch), np.asarray(pose_q_batch)]
