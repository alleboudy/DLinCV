import numpy as np
import random
import cv2
#import caffe
import tensorflow as tf
import os


directory = "/usr/prakt/w065/posenet/KingsCollege/"

dataset = 'dataset_train.txt'
outputDirectory = "/usr/prakt/w065/posenet/TFData/"


poses = [] #will contain poses followed by qs
images = []

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# images and labels array as input
def convert_to_TFRecord(images, poses, outputDirectory):
    num_examples = len(poses)
    if len(images) != num_examples:
        raise ValueError("Images size %d does not match label size %d." %
                     (len(images), num_examples))
    rows = 224
    cols = 224
    depth = 3
    id='tfrecords'
    filename = os.path.join( outputDirectory+ id + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in r:
        img = cv2.imread(images[index])
        #img = cv2.resize(img, (224,224))    # to reproduce PoseNet results, please resize the images so that the shortest side is 256 pixels
        #img = np.transpose(img,(2,0,1))
        image_raw = img.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=poses[index])),
        'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())


def ResizeCropImage(image):
    # we need to keep in mind aspect ratio so the image does
    # not look skewed or distorted -- therefore, we calculate
    # the ratio of the new image to the old image
    r = 256.0 / image.shape[0]
    dim = ( int(image.shape[1] * r),256)

    # perform the actual resizing of the image and show it
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)[0:224, 0:224]
    #cv2.imshow("resized", resized)
    #cv2.waitKey(0)


#MAiN

with open(directory+dataset) as f:
    next(f)  # skip the 3 header lines
    next(f)
    next(f)
    for line in f:
        fname, p0,p1,p2,p3,p4,p5,p6 = line.split()
        p0 = float(p0)
        p1 = float(p1)
        p2 = float(p2)
        p3 = float(p3)
        p4 = float(p4)
        p5 = float(p5)
        p6 = float(p6)
        poses.append((p0,p1,p2,p3,p4,p5,p6))
        images.append(directory+fname)

r = list(range(len(images)))
random.shuffle(r)
convert_to_TFRecord(images,poses,outputDirectory)



