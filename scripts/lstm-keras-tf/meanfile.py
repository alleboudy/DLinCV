import caffe
import numpy as np
import sys

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( 'imagemean.binaryproto', 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
print ('out shape:')
print ( out.shape)
print ('arr shape:')
print (arr.shape)

