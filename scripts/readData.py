
from PIL import Image
import numpy as np
import tensorflow as tf

def read_and_decode(filename_queue):
 reader = tf.TFRecordReader()
 _, serialized_example = reader.read(filename_queue)
 features = tf.parse_single_example(
  serialized_example,
  # Defaults are not specified since both keys are required.
  features={
      'image_raw': tf.FixedLenFeature([], tf.string),
      'label': tf.VarLenFeature(tf.float32),
      'height': tf.FixedLenFeature([], tf.int64),
      'width': tf.FixedLenFeature([], tf.int64),
      'depth': tf.FixedLenFeature([], tf.int64)
  })
 image = tf.decode_raw(features['image_raw'], tf.uint8)
 label = tf.cast(features['label'], tf.float32)
 height = tf.cast(features['height'], tf.int32)
 width = tf.cast(features['width'], tf.int32)
 depth = tf.cast(features['depth'], tf.int32)
 return image, label, height, width, depth


def get_all_records(FILE):
 with tf.Session() as sess:
   filename_queue = tf.train.string_input_producer([ FILE ])
   image,label, height, width, depth = read_and_decode(filename_queue)
   image = tf.reshape(image, tf.pack([height, width, 3]))
   image.set_shape([224,224,3])
   init_op = tf.initialize_all_variables()
   sess.run(init_op)
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(coord=coord)
   for i in range(2):#calling this area multiple times to get examples
     example, l = sess.run([image, label])
     img = Image.fromarray(example, 'RGB')
     img.save( "output/" + str(i) + '-train.png')
     #img = Image.fromarray(example, 'RGB')
     #img.save( "output/" + str(i) + '-train.png')
     print 'label:'
     print l.values
     print 'example'
     print (example[0])
   coord.request_stop()
   coord.join(threads)

get_all_records("/usr/prakt/w065/posenet/TFData/tfrecords.tfrecords")
