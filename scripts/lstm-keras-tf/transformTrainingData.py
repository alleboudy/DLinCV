import cv2

directory = '/usr/prakt/w065/posenet/KingsCollege/'
dataset = 'dataset_train.txt'

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

    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)[0:224, 0:224]



with open(directory+dataset) as f:
    next(f)  # skip the 3 header lines
    next(f)
    next(f)
    for line in f:
        fname, p0,p1,p2,p3,p4,p5,p6 = line.split()
        img = cv2.imread(directory+fname)
        img = ResizeCropImage(img)  
        cv2.imwrite(directory+fname, img)


