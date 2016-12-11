from scipy.misc import imread, imresize

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from googlenet_custom_layers import PoolHelper,LRN


def create_posenet(weights_path=None):
    # creates GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    
    input = Input(shape=(3, 224, 224))
    
    conv1 = Convolution2D(64,7,7,subsample=(2,2),border_mode='same',activation='relu',name='conv1',W_regularizer=l2(0.0002))(input)
    
    conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1)
    
    pool1_helper = PoolHelper()(conv1_zero_pad)
    
    pool1 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool1')(pool1_helper)
    
    norm1 = LRN(name='norm1')(pool1)
    
    reduction2 = Convolution2D(64,1,1,border_mode='same',activation='relu',name='reduction2',W_regularizer=l2(0.0002))(norm1)
    
    conv2 = Convolution2D(192,3,3,border_mode='same',activation='relu',name='conv2',W_regularizer=l2(0.0002))(reduction2)
    
    norm2 = LRN(name='conv2')(conv2)
    
    conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(norm2)
    
    pool2_helper = PoolHelper()(conv2_zero_pad)
    
    pool2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool2')(pool2_helper)
    
    
    icp1_out0 = Convolution2D(64,1,1,border_mode='same',activation='relu',name='icp1_out0',W_regularizer=l2(0.0002))(pool2)
    
    icp1_reduction1 = Convolution2D(96,1,1,border_mode='same',activation='relu',name='icp1_reduction1',W_regularizer=l2(0.0002))(pool2)
    
    icp1_out1 = Convolution2D(128,3,3,border_mode='same',activation='relu',name='icp1_out1',W_regularizer=l2(0.0002))(icp1_reduction1)
    
    icp1_reduction2 = Convolution2D(16,1,1,border_mode='same',activation='relu',name='icp1_reduction2',W_regularizer=l2(0.0002))(pool2)
    
    icp1_out2 = Convolution2D(32,5,5,border_mode='same',activation='relu',name='icp1_out2',W_regularizer=l2(0.0002))(icp1_reduction2)
    
    icp1_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp1_pool')(pool2)
    
    icp1_out3 = Convolution2D(32,1,1,border_mode='same',activation='relu',name='icp1_out3',W_regularizer=l2(0.0002))(icp1_pool)
    
    icp2_in = merge([icp1_out0,icp1_out1,icp1_out2,icp1_out3],mode='concat',concat_axis=1,name='icp2_in')
    
    
    icp2_out0 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='icp2_out0',W_regularizer=l2(0.0002))(icp2_in)
    
    icp2_reduction1 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='icp2_reduction1',W_regularizer=l2(0.0002))(icp2_in)
    
    icp2_out1 = Convolution2D(192,3,3,border_mode='same',activation='relu',name='icp2_out1',W_regularizer=l2(0.0002))(icp2_reduction1)
    
    icp2_reduction2 = Convolution2D(32,1,1,border_mode='same',activation='relu',name='icp2_reduction2',W_regularizer=l2(0.0002))(icp2_in)
    
    icp2_out2 = Convolution2D(96,5,5,border_mode='same',activation='relu',name='icp2_out2',W_regularizer=l2(0.0002))(icp2_reduction2)
    
    icp2_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp2_pool')(icp2_in)
    
    icp2_out3 = Convolution2D(64,1,1,border_mode='same',activation='relu',name='icp2_out3',W_regularizer=l2(0.0002))(icp2_pool)
    
    icp2_out = merge([icp2_out0,icp2_out1,icp2_out2,icp2_out3],mode='concat',concat_axis=1,name='icp2_out')
    
    
    icp2_out_zero_pad = ZeroPadding2D(padding=(1, 1))(icp2_out)
    
    pool3_helper = PoolHelper()(icp2_out_zero_pad)
    
    icp3_in = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='icp3_in')(pool3_helper)
    
    
    icp3_out0 = Convolution2D(192,1,1,border_mode='same',activation='relu',name='icp3_out0',W_regularizer=l2(0.0002))(icp3_in)
    
    icp3_reduction1 = Convolution2D(96,1,1,border_mode='same',activation='relu',name='icp3_reduction1',W_regularizer=l2(0.0002))(icp3_in)
    
    icp3_out1 = Convolution2D(208,3,3,border_mode='same',activation='relu',name='icp3_out1',W_regularizer=l2(0.0002))(icp3_reduction1)
    
    icp3_reduction2 = Convolution2D(16,1,1,border_mode='same',activation='relu',name='icp3_reduction2',W_regularizer=l2(0.0002))(icp3_in)
    
    icp3_out2 = Convolution2D(48,5,5,border_mode='same',activation='relu',name='icp3_out2',W_regularizer=l2(0.0002))(icp3_reduction2)
    
    icp3_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp3_pool')(icp3_in)
    
    icp3_out3 = Convolution2D(64,1,1,border_mode='same',activation='relu',name='icp3_out3',W_regularizer=l2(0.0002))(icp3_pool)
    
    icp3_out = merge([icp3_out0,icp3_out1,icp3_out2,icp3_out3],mode='concat',concat_axis=1,name='icp3_out')
    
    
    cls1_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),name='cls1_pool')(icp3_out)
    
    cls1_reduction_pose = Convolution2D(128,1,1,border_mode='same',activation='relu',name='cls1_reduction_pose',W_regularizer=l2(0.0002))(cls1_pool)
    
    loss1_flat = Flatten()(cls1_reduction_pose)
    
    cls1_fc1_pose = Dense(1024,activation='relu',name='cls1_fc1_pose',W_regularizer=l2(0.0002))(loss1_flat)
    
    cls1_drop = Dropout(0.7)(cls1_fc1_pose)
    
    cls1_fc_pose_xyz = Dense(3,name='cls1_fc_pose_xyz',W_regularizer=l2(0.0002))(cls1_drop)

    cls1_fc_pose_wpqr = Dense(4,name='cls1_fc_pose_wpqr',W_regularizer=l2(0.0002))(cls1_drop)
    
    
    icp4_out0 = Convolution2D(160,1,1,border_mode='same',activation='relu',name='icp4_out0',W_regularizer=l2(0.0002))(icp3_out)
    
    icp4_reduction1 = Convolution2D(112,1,1,border_mode='same',activation='relu',name='icp4_reduction1',W_regularizer=l2(0.0002))(icp3_out)
    
    icp4_out1 = Convolution2D(224,3,3,border_mode='same',activation='relu',name='icp4_out1',W_regularizer=l2(0.0002))(icp4_reduction1)
    
    icp4_reduction2 = Convolution2D(24,1,1,border_mode='same',activation='relu',name='icp4_reduction2',W_regularizer=l2(0.0002))(icp3_out)
    
    icp4_out2 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='icp4_out2',W_regularizer=l2(0.0002))(icp4_reduction2)
    
    icp4_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp4_pool')(icp3_out)
    
    icp4_out3 = Convolution2D(64,1,1,border_mode='same',activation='relu',name='icp4_out3',W_regularizer=l2(0.0002))(icp4_pool)
    
    icp4_out = merge([icp4_out0,icp4_out1,icp4_out2,icp4_out3],mode='concat',concat_axis=1,name='icp4_out')
    
    
    icp5_out0 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='icp5_out0',W_regularizer=l2(0.0002))(icp4_out)
    
    icp5_reduction1 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='icp5_reduction1',W_regularizer=l2(0.0002))(icp4_out)
    
    icp5_out1 = Convolution2D(256,3,3,border_mode='same',activation='relu',name='icp5_out1',W_regularizer=l2(0.0002))(icp5_reduction1)
    
    icp5_reduction2 = Convolution2D(24,1,1,border_mode='same',activation='relu',name='icp5_reduction2',W_regularizer=l2(0.0002))(icp4_out)
    
    icp5_out2 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='icp5_out2',W_regularizer=l2(0.0002))(icp5_reduction2)
    
    icp5_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp5_pool')(icp4_out)
    
    icp5_out3 = Convolution2D(64,1,1,border_mode='same',activation='relu',name='icp5_out3',W_regularizer=l2(0.0002))(icp5_pool)
    
    icp5_out = merge([icp5_out0,icp5_out1,icp5_out2,icp5_out3],mode='concat',concat_axis=1,name='icp5_out')
    
    
    icp6_out0 = Convolution2D(112,1,1,border_mode='same',activation='relu',name='icp6_out0',W_regularizer=l2(0.0002))(icp5_out)
    
    icp6_reduction1 = Convolution2D(144,1,1,border_mode='same',activation='relu',name='icp6_reduction1',W_regularizer=l2(0.0002))(icp5_out)
    
    icp6_out1 = Convolution2D(288,3,3,border_mode='same',activation='relu',name='icp6_out1',W_regularizer=l2(0.0002))(icp6_reduction1)
    
    icp6_reduction2 = Convolution2D(32,1,1,border_mode='same',activation='relu',name='icp6_reduction2',W_regularizer=l2(0.0002))(icp5_out)
    
    icp6_out2 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='icp6_out2',W_regularizer=l2(0.0002))(icp6_reduction2)
    
    icp6_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp6_pool')(icp5_out)
    
    icp6_out3 = Convolution2D(64,1,1,border_mode='same',activation='relu',name='icp6_out3',W_regularizer=l2(0.0002))(icp6_pool)
    
    icp6_out = merge([icp6_out0,icp6_out1,icp6_out2,icp6_out3],mode='concat',concat_axis=1,name='icp6_out')
    
    
    cls2_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),name='cls2_pool')(icp6_out)
    
    cls2_reduction_pose = Convolution2D(128,1,1,border_mode='same',activation='relu',name='cls2_reduction',W_regularizer=l2(0.0002))(cls2_pool)
    
    loss2_flat = Flatten()(cls2_reduction_pose)
    
    cls2_fc1 = Dense(1024,activation='relu',name='cls2_fc1',W_regularizer=l2(0.0002))(loss2_flat)
    
    cls2_drop = Dropout(0.7)(cls2_fc1)
    
    cls2_fc_pose_xyz = Dense(3,name='cls2_fc_pose_xyz',W_regularizer=l2(0.0002))(cls2_drop)
    
    cls2_fc_pose_wpqr = Dense(4,name='cls2_fc_pose_wpqr',W_regularizer=l2(0.0002))(cls2_drop)
    
    
    icp7_out0 = Convolution2D(256,1,1,border_mode='same',activation='relu',name='icp7_out0',W_regularizer=l2(0.0002))(icp6_out)
    
    icp7_reduction1 = Convolution2D(160,1,1,border_mode='same',activation='relu',name='icp7_reduction1',W_regularizer=l2(0.0002))(icp6_out)
    
    icp7_out1 = Convolution2D(320,3,3,border_mode='same',activation='relu',name='icp7_out1',W_regularizer=l2(0.0002))(icp7_reduction1)
    
    icp7_reduction2 = Convolution2D(32,1,1,border_mode='same',activation='relu',name='icp7_reduction2',W_regularizer=l2(0.0002))(icp6_out)
    
    icp7_out2 = Convolution2D(128,5,5,border_mode='same',activation='relu',name='icp7_out2',W_regularizer=l2(0.0002))(icp7_reduction2)
    
    icp7_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp7_pool')(icp6_out)
    
    icp7_out3 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='icp7_out3',W_regularizer=l2(0.0002))(icp7_pool)
    
    icp7_out = merge([icp7_out0,icp7_out1,icp7_out2,icp7_out3],mode='concat',concat_axis=1,name='icp7_out')
    
    
    icp7_out_zero_pad = ZeroPadding2D(padding=(1, 1))(icp7_out)
    
    pool4_helper = PoolHelper()(icp7_out_zero_pad)
    
    icp8_in = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='icp8_in')(pool4_helper)
    
    
    icp8_out0 = Convolution2D(256,1,1,border_mode='same',activation='relu',name='icp8_out0',W_regularizer=l2(0.0002))(icp8_in)
    
    icp8_reduction1 = Convolution2D(160,1,1,border_mode='same',activation='relu',name='icp8_reduction1',W_regularizer=l2(0.0002))(icp8_in)
    
    icp8_out1 = Convolution2D(320,3,3,border_mode='same',activation='relu',name='icp8_out1',W_regularizer=l2(0.0002))(icp8_reduction1)
    
    icp8_reduction2 = Convolution2D(32,1,1,border_mode='same',activation='relu',name='icp8_reduction2',W_regularizer=l2(0.0002))(icp8_in)
    
    icp8_out2 = Convolution2D(128,5,5,border_mode='same',activation='relu',name='icp8_out2',W_regularizer=l2(0.0002))(icp8_reduction2)
    
    icp8_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp8_pool')(icp8_in)
    
    icp8_out3 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='icp8_out3',W_regularizer=l2(0.0002))(icp8_pool)
    
    icp8_out = merge([icp8_out0,icp8_out1,icp8_out2,icp8_out3],mode='concat',concat_axis=1,name='icp8_out')
    
    icp9_out0 = Convolution2D(384,1,1,border_mode='same',activation='relu',name='icp9_out0',W_regularizer=l2(0.0002))(icp8_out)
    
    icp9_reduction1 = Convolution2D(192,1,1,border_mode='same',activation='relu',name='icp9_reduction1',W_regularizer=l2(0.0002))(icp8_out)
    
    icp9_out1 = Convolution2D(384,3,3,border_mode='same',activation='relu',name='icp9_out1',W_regularizer=l2(0.0002))(icp9_reduction1)
    
    icp9_reduction2 = Convolution2D(48,1,1,border_mode='same',activation='relu',name='icp9_reduction2',W_regularizer=l2(0.0002))(icp8_out)
    
    icp9_out2 = Convolution2D(128,5,5,border_mode='same',activation='relu',name='icp9_out2',W_regularizer=l2(0.0002))(icp9_reduction2)
    
    icp9_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp9_pool')(icp8_out)
    
    icp9_out3 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='icp3_out3',W_regularizer=l2(0.0002))(icp9_pool)
    
    icp9_out = merge([icp9_out0,icp9_out1,icp9_out2,icp9_out3],mode='concat',concat_axis=1,name='icp9_out')
    
    cls3_pool = AveragePooling2D(pool_size=(7,7),strides=(1,1),name='cls3_pool')(icp9_out)
    
    loss3_flat = Flatten()(cls3_pool)
    
    cls3_fc1_pose = Dense(2048,activation='relu',name='cls3_fc1_pose',W_regularizer=l2(0.0002))(loss1_flat)
    
    cls3_fc1 = Dropout(0.5)(cls3_fc1_pose)
    
    cls3_fc_pose_xyz = Dense(3,name='cls3_fc_pose_xyz',W_regularizer=l2(0.0002))(cls3_fc1)
    
    cls3_fc_pose_wpqr = Dense(4,name='cls3_fc_pose_wpqr',W_regularizer=l2(0.0002))(cls3_fc1)
    
    googlenet = Model(input=input, output=[cls1_fc_pose_xyz,cls2_fc_pose_xyz,cls3_fc_pose_xyz,cls1_fc_pose_wpqr,cls2_fc_pose_wpqr,cls3_fc_pose_wpqr])
    
    if weights_path:
        googlenet.load_weights(weights_path)
    
    return googlenet



if __name__ == "__main__":
    img = imresize(imread('cat.jpg', mode='RGB'), (224, 224)).astype(np.float32)
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    # Test pretrained model
    model = create_googlenet('googlenet_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(img) # note: the model has three outputs
    print np.argmax(out[2])
