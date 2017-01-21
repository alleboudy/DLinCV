from scipy.misc import imread, imresize
from posenet import posenet

img = imresize(imread('cat.jpg' ), (224, 224)).astype(np.float32)
img[:, :, 0] -= 123.68
img[:, :, 1] -= 116.779
img[:, :, 2] -= 103.939
img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
img = img.transpose((2, 0, 1))
img = np.expand_dims(img, axis=0)

predictPose('mergedweights.h5',img)
