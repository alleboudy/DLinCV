from scipy.misc import imread, imresize
import numpy as np
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation, LSTM,BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
#from keras.regularizers import l2
import settings

from keras import backend as K

#K.set_learning_phase(1) #set learning phase

def create_cnn_lstm(weights_path=None):
    
    input = Input(shape=(settings.stepSize, 3, 224, 224))
    
    conv1_7x7_s2 = TimeDistributed(Convolution2D(64,7,7,subsample=(2,2),border_mode='same',activation='relu'),name='conv1')(input)
    
    #conv1_zero_pad = TimeDistributed(ZeroPadding2D(padding=(1, 1)))(conv1_7x7_s2)
    
    #pool1_helper = conv1_zero_pad
    
    pool1_3x3_s2 = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='same'),name='pool1')(conv1_7x7_s2)
    
    pool1_norm1 = TimeDistributed( BatchNormalization(axis=2),name='norm1')(pool1_3x3_s2)
    
    conv2_3x3_reduce = TimeDistributed(Convolution2D(64,1,1,border_mode='same',activation='relu'),name='reduction2')(pool1_norm1)
    
    conv2_3x3 = TimeDistributed(Convolution2D(192,3,3,border_mode='same',activation='relu'),name='conv2')(conv2_3x3_reduce)
    
    conv2_norm2 = TimeDistributed(BatchNormalization(axis=2),name='norm2')(conv2_3x3)
    
    #conv2_zero_pad = TimeDistributed(ZeroPadding2D(padding=(1, 1)))(conv2_norm2)
    
    #pool2_helper = conv2_zero_pad
    
    pool2_3x3_s2 = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid'),name='pool2')(conv2_norm2)
    
    
    inception_3a_1x1 = TimeDistributed(Convolution2D(64,1,1,border_mode='same',activation='relu'),name='icp1_out0')(pool2_3x3_s2)
    
    inception_3a_3x3_reduce = TimeDistributed(Convolution2D(96,1,1,border_mode='same',activation='relu'),name='icp1_reduction1')(pool2_3x3_s2)
    
    inception_3a_3x3 = TimeDistributed(Convolution2D(128,3,3,border_mode='same',activation='relu'),name='icp1_out1')(inception_3a_3x3_reduce)
    
    inception_3a_5x5_reduce = TimeDistributed(Convolution2D(16,1,1,border_mode='same',activation='relu'),name='icp1_reduction2')(pool2_3x3_s2)
    
    inception_3a_5x5 = TimeDistributed(Convolution2D(32,5,5,border_mode='same',activation='relu'),name='icp1_out2')(inception_3a_5x5_reduce)
    
    inception_3a_pool = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same'),name='icp1_pool')(pool2_3x3_s2)
    
    inception_3a_pool_proj = TimeDistributed(Convolution2D(32,1,1,border_mode='same',activation='relu'),name='icp1_out3')(inception_3a_pool)
    
    inception_3a_output = merge([inception_3a_1x1,inception_3a_3x3,inception_3a_5x5,inception_3a_pool_proj],mode='concat',concat_axis=2,name='icp2_in')
    
    
    inception_3b_1x1 = TimeDistributed(Convolution2D(128,1,1,border_mode='same',activation='relu'),name='icp2_reduction1')(inception_3a_output)
    
    inception_3b_3x3_reduce = TimeDistributed(Convolution2D(128,1,1,border_mode='same',activation='relu'),name='icp2_out0')(inception_3a_output)
    
    inception_3b_3x3 = TimeDistributed(Convolution2D(192,3,3,border_mode='same',activation='relu'),name='icp2_out1')(inception_3b_3x3_reduce)
    
    inception_3b_5x5_reduce = TimeDistributed(Convolution2D(32,1,1,border_mode='same',activation='relu'),name='icp2_reduction2')(inception_3a_output)
    
    inception_3b_5x5 = TimeDistributed(Convolution2D(96,5,5,border_mode='same',activation='relu'),name='icp2_out2')(inception_3b_5x5_reduce)
    
    inception_3b_pool = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same'),name='icp2_pool')(inception_3a_output)
    
    inception_3b_pool_proj = TimeDistributed(Convolution2D(64,1,1,border_mode='same',activation='relu'),name='icp2_out3')(inception_3b_pool)
    
    inception_3b_output = merge([inception_3b_1x1,inception_3b_3x3,inception_3b_5x5,inception_3b_pool_proj],mode='concat',concat_axis=2,name='icp2_out')
    
    
    #inception_3b_output_zero_pad = TimeDistributed(ZeroPadding2D(padding=(1, 1)))(inception_3b_output)
    
    #pool3_helper = inception_3b_output_zero_pad #PoolHelper()(inception_3b_output_zero_pad)
    
    pool3_3x3_s2 = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='same'),name='icp3_in')(inception_3b_output)
    
    
    inception_4a_1x1 = TimeDistributed(Convolution2D(192,1,1,border_mode='same',activation='relu'),name='icp3_out0')(pool3_3x3_s2)
    
    inception_4a_3x3_reduce = TimeDistributed(Convolution2D(96,1,1,border_mode='same',activation='relu'),name='icp3_reduction1')(pool3_3x3_s2)
    
    inception_4a_3x3 = TimeDistributed(Convolution2D(208,3,3,border_mode='same',activation='relu'),name='icp3_out1')(inception_4a_3x3_reduce)
    
    inception_4a_5x5_reduce = TimeDistributed(Convolution2D(16,1,1,border_mode='same',activation='relu'),name='icp3_reduction2')(pool3_3x3_s2)
    
    inception_4a_5x5 = TimeDistributed(Convolution2D(48,5,5,border_mode='same',activation='relu'),name='icp3_out2')(inception_4a_5x5_reduce)
    
    inception_4a_pool = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same'),name='icp3_pool')(pool3_3x3_s2)
    
    inception_4a_pool_proj = TimeDistributed(Convolution2D(64,1,1,border_mode='same',activation='relu'),name='icp3_out3')(inception_4a_pool)
    
    inception_4a_output = merge([inception_4a_1x1,inception_4a_3x3,inception_4a_5x5,inception_4a_pool_proj],mode='concat',concat_axis=2,name='icp3_out')
    
    
    loss1_ave_pool = TimeDistributed(AveragePooling2D(pool_size=(5,5),strides=(3,3),border_mode='valid'),name='cls1_pool')(inception_4a_output)
    
    loss1_conv = TimeDistributed(Convolution2D(128,1,1,border_mode='same',activation='relu'),name='cls1_reduction_pose')(loss1_ave_pool)
    
    loss1_flat = TimeDistributed(Flatten())(loss1_conv)
    
    loss1_fc = TimeDistributed(Dense(1024,activation='relu'),name='cls1_fc1_pose')(loss1_flat)
    
    #loss1_drop_fc = TimeDistributed(Dropout(0.7))(loss1_fc)

    # cls1_fc_pose_xyz = TimeDistributed(Dense(3,name='cls1_fc_pose_xyz'))(loss1_drop_fc)

    # cls1_fc_pose_wpqr = TimeDistributed(Dense(4,name='cls1_fc_pose_wpqr'))(loss1_drop_fc)
    lstma = LSTM(512 ,return_sequences=True, input_shape=(settings.stepSize,1024))(loss1_fc)
    cls1_fc_pose_xyz = TimeDistributed(Dense(3),name='cls1_fc_pose_xyz')(lstma)

    cls1_fc_pose_wpqr = TimeDistributed(Dense(4),name='cls1_fc_pose_wpqr')(lstma)
    
   # loss1_classifier = TimeDistributed(Dense(1000,name='loss1/classifier'))(loss1_drop_fc)
    
   # loss1_classifier_act = TimeDistributed(Activation('softmax'))(loss1_classifier)
    
    
    inception_4b_1x1 = TimeDistributed(Convolution2D(160,1,1,border_mode='same',activation='relu'),name='icp4_out0')(inception_4a_output)
    
    inception_4b_3x3_reduce = TimeDistributed(Convolution2D(112,1,1,border_mode='same',activation='relu'),name='icp4_reduction1')(inception_4a_output)
    
    inception_4b_3x3 = TimeDistributed(Convolution2D(224,3,3,border_mode='same',activation='relu'),name='icp4_out1')(inception_4b_3x3_reduce)
    
    inception_4b_5x5_reduce = TimeDistributed(Convolution2D(24,1,1,border_mode='same',activation='relu'),name='icp4_reduction2')(inception_4a_output)
    
    inception_4b_5x5 = TimeDistributed(Convolution2D(64,5,5,border_mode='same',activation='relu'),name='icp4_out2')(inception_4b_5x5_reduce)
    
    inception_4b_pool = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same'),name='icp4_pool')(inception_4a_output)
    
    inception_4b_pool_proj = TimeDistributed(Convolution2D(64,1,1,border_mode='same',activation='relu'),name='icp4_out3')(inception_4b_pool)
    
    inception_4b_output = merge([inception_4b_1x1,inception_4b_3x3,inception_4b_5x5,inception_4b_pool_proj],mode='concat',concat_axis=2,name='icp4_out')
    
    
    inception_4c_1x1 = TimeDistributed(Convolution2D(128,1,1,border_mode='same',activation='relu'),name='icp5_out0')(inception_4b_output)
    
    inception_4c_3x3_reduce = TimeDistributed(Convolution2D(128,1,1,border_mode='same',activation='relu'),name='icp5_reduction1')(inception_4b_output)
    
    inception_4c_3x3 = TimeDistributed(Convolution2D(256,3,3,border_mode='same',activation='relu'),name='icp5_out1')(inception_4c_3x3_reduce)
    
    inception_4c_5x5_reduce = TimeDistributed(Convolution2D(24,1,1,border_mode='same',activation='relu'),name='icp5_reduction2')(inception_4b_output)
    
    inception_4c_5x5 = TimeDistributed(Convolution2D(64,5,5,border_mode='same',activation='relu'),name='icp5_out2')(inception_4c_5x5_reduce)
    
    inception_4c_pool = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same'),name='icp5_pool')(inception_4b_output)
    
    inception_4c_pool_proj = TimeDistributed(Convolution2D(64,1,1,border_mode='same',activation='relu'),name='icp5_out3')(inception_4c_pool)
    
    inception_4c_output = merge([inception_4c_1x1,inception_4c_3x3,inception_4c_5x5,inception_4c_pool_proj],mode='concat',concat_axis=2,name='icp5_out')
    
    
    inception_4d_1x1 = TimeDistributed(Convolution2D(112,1,1,border_mode='same',activation='relu'),name='icp6_out0')(inception_4c_output)
    
    inception_4d_3x3_reduce = TimeDistributed(Convolution2D(144,1,1,border_mode='same',activation='relu'),name='icp6_reduction1')(inception_4c_output)
    
    inception_4d_3x3 = TimeDistributed(Convolution2D(288,3,3,border_mode='same',activation='relu'),name='icp6_out1')(inception_4d_3x3_reduce)
    
    inception_4d_5x5_reduce = TimeDistributed(Convolution2D(32,1,1,border_mode='same',activation='relu'),name='icp6_reduction2')(inception_4c_output)
    
    inception_4d_5x5 = TimeDistributed(Convolution2D(64,5,5,border_mode='same',activation='relu'),name='icp6_out2')(inception_4d_5x5_reduce)
    
    inception_4d_pool = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same'),name='icp6_pool')(inception_4c_output)
    
    inception_4d_pool_proj = TimeDistributed(Convolution2D(64,1,1,border_mode='same',activation='relu'),name='icp6_out3')(inception_4d_pool)
    
    inception_4d_output = merge([inception_4d_1x1,inception_4d_3x3,inception_4d_5x5,inception_4d_pool_proj],mode='concat',concat_axis=2,name='icp6_out')
    
    
    loss2_ave_pool = TimeDistributed(AveragePooling2D(pool_size=(5,5),strides=(3,3),border_mode='valid'),name='cls2_pool')(inception_4d_output)
    
    loss2_conv = TimeDistributed(Convolution2D(128,1,1,border_mode='same',activation='relu'),name='cls2_reduction_pose')(loss2_ave_pool)
    
    loss2_flat = TimeDistributed(Flatten())(loss2_conv)
    
    loss2_fc = TimeDistributed(Dense(1024,activation='relu'),name='cls2_fc1')(loss2_flat)
    
    #loss2_drop_fc = TimeDistributed(Dropout(0.7))(loss2_fc)

    # cls2_fc_pose_xyz = TimeDistributed(Dense(3,name='cls2_fc_pose_xyz'))(loss2_drop_fc)

    # cls2_fc_pose_wpqr = TimeDistributed(Dense(4,name='cls2_fc_pose_wpqr'))(loss2_drop_fc)

    #cls3_fc2 = TimeDistributed(Dropout(0.5))(cls3_out)
    lstmb = LSTM(512 ,return_sequences=True, input_shape=(settings.stepSize,1024))(loss2_fc)
    cls2_fc_pose_xyz = TimeDistributed(Dense(3),name='cls2_fc_pose_xyz')(lstmb)

    cls2_fc_pose_wpqr = TimeDistributed(Dense(4),name='cls2_fc_pose_wpqr')(lstmb)


    
    # loss2_classifier = TimeDistributed(Dense(1000,name='loss2/classifier'))(loss2_drop_fc)
    
    # loss2_classifier_act = TimeDistributed(Activation('softmax'))(loss2_classifier)
    
    
    inception_4e_1x1 = TimeDistributed(Convolution2D(256,1,1,border_mode='same',activation='relu'),name='icp7_out0')(inception_4d_output)
    
    inception_4e_3x3_reduce = TimeDistributed(Convolution2D(160,1,1,border_mode='same',activation='relu'),name='icp7_reduction1')(inception_4d_output)
    
    inception_4e_3x3 = TimeDistributed(Convolution2D(320,3,3,border_mode='same',activation='relu'),name='icp7_out1')(inception_4e_3x3_reduce)
    
    inception_4e_5x5_reduce = TimeDistributed(Convolution2D(32,1,1,border_mode='same',activation='relu'),name='icp7_reduction2')(inception_4d_output)
    
    inception_4e_5x5 = TimeDistributed(Convolution2D(128,5,5,border_mode='same',activation='relu'),name='icp7_out2')(inception_4e_5x5_reduce)
    
    inception_4e_pool = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same'),name='icp7_pool')(inception_4d_output)
    
    inception_4e_pool_proj = TimeDistributed(Convolution2D(128,1,1,border_mode='same',activation='relu'),name='icp7_out3')(inception_4e_pool)
    
    inception_4e_output = merge([inception_4e_1x1,inception_4e_3x3,inception_4e_5x5,inception_4e_pool_proj],mode='concat',concat_axis=2,name='icp7_out')
    
    
    #inception_4e_output_zero_pad = TimeDistributed(ZeroPadding2D(padding=(1, 1)))(inception_4e_output)
    
    #pool4_helper = inception_4e_output_zero_pad #PoolHelper()(inception_4e_output_zero_pad)
    
    pool4_3x3_s2 = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='same'),name='icp8_in')(inception_4e_output)
    
    
    inception_5a_1x1 = TimeDistributed(Convolution2D(256,1,1,border_mode='same',activation='relu'),name='icp8_out0')(pool4_3x3_s2)
    
    inception_5a_3x3_reduce = TimeDistributed(Convolution2D(160,1,1,border_mode='same',activation='relu'),name='icp8_reduction1')(pool4_3x3_s2)
    
    inception_5a_3x3 = TimeDistributed(Convolution2D(320,3,3,border_mode='same',activation='relu'),name='icp8_out1')(inception_5a_3x3_reduce)
    
    inception_5a_5x5_reduce = TimeDistributed(Convolution2D(32,1,1,border_mode='same',activation='relu'),name='icp8_reduction2')(pool4_3x3_s2)
    
    inception_5a_5x5 = TimeDistributed(Convolution2D(128,5,5,border_mode='same',activation='relu'),name='icp8_out2')(inception_5a_5x5_reduce)
    
    inception_5a_pool = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same'),name='icp8_pool')(pool4_3x3_s2)
    
    inception_5a_pool_proj = TimeDistributed(Convolution2D(128,1,1,border_mode='same',activation='relu'),name='icp8_out3')(inception_5a_pool)
    
    inception_5a_output = merge([inception_5a_1x1,inception_5a_3x3,inception_5a_5x5,inception_5a_pool_proj],mode='concat',concat_axis=2,name='icp8_out')
    
    
    inception_5b_1x1 = TimeDistributed(Convolution2D(384,1,1,border_mode='same',activation='relu'),name='icp9_out0')(inception_5a_output)
    
    inception_5b_3x3_reduce = TimeDistributed(Convolution2D(192,1,1,border_mode='same',activation='relu'),name='icp9_reduction1')(inception_5a_output)
    
    inception_5b_3x3 = TimeDistributed(Convolution2D(384,3,3,border_mode='same',activation='relu'),name='icp9_out1')(inception_5b_3x3_reduce)
    
    inception_5b_5x5_reduce = TimeDistributed(Convolution2D(48,1,1,border_mode='same',activation='relu'),name='icp9_reduction2')(inception_5a_output)
    
    inception_5b_5x5 = TimeDistributed(Convolution2D(128,5,5,border_mode='same',activation='relu'),name='icp9_out2')(inception_5b_5x5_reduce)
    
    inception_5b_pool = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same'),name='icp9_pool')(inception_5a_output)
    
    inception_5b_pool_proj = TimeDistributed(Convolution2D(128,1,1,border_mode='same',activation='relu'),name='icp9_out3')(inception_5b_pool)
    
    inception_5b_output = merge([inception_5b_1x1,inception_5b_3x3,inception_5b_5x5,inception_5b_pool_proj],mode='concat',concat_axis=2,name='icp9_out')
    
    
    pool5_7x7_s1 = TimeDistributed(AveragePooling2D(pool_size=(7,7),strides=(1,1),border_mode='valid'),name='cls3_pool')(inception_5b_output)
    
    loss3_flat = TimeDistributed(Flatten())(pool5_7x7_s1)

    cls3_fc1_pose = TimeDistributed(Dense(2048,activation='relu',init="normal"),name='cls3_fc1_pose')(loss3_flat)

    #cls3_fc1 = TimeDistributed(Dropout(0.5))(cls3_fc1_pose)

    cls3_out = TimeDistributed(Dense(1024,activation='relu'),name='cls3_out')(cls3_fc1_pose)
    #cls3_fc2 = TimeDistributed(Dropout(0.5))(cls3_out)
    lstm1 = LSTM(512 ,return_sequences=True, input_shape=(settings.stepSize,1024))(cls3_out)
    pose_xyz = TimeDistributed(Dense(3),name='pose_xyz')(lstm1)

    pose_wpqr = TimeDistributed(Dense(4),name='pose_wpqr')(lstm1)



  #  cls4_out = TimeDistributed(Dense(512,activation='relu',name='cls4_out'))(cls3_fc2)
   # cls4_fc2 = TimeDistributed(Dropout(0.5))(cls4_out)

 #   cls5_out = TimeDistributed(Dense(128,activation='relu',name='cls5_out'))(cls4_fc2)
   # cls5_fc2 = TimeDistributed(Dropout(0.5))(cls5_out)


#    cls3_fc_pose_xyz = TimeDistributed(Dense(3,name='cls3_fc_pose_xyz'))(cls3_fc1)

#    cls3_fc_pose_wpqr = TimeDistributed(Dense(4,name='cls3_fc_pose_wpqr'))(cls3_fc1)
    
#    pool5_drop_7x7_s1 = TimeDistributed(Dropout(0.4))(loss3_flat)
    
#    loss3_classifier = TimeDistributed(Dense(1000,name='loss3/classifier'))(pool5_drop_7x7_s1)
    
#    loss3_classifier_act = TimeDistributed(Activation('softmax',name='prob'))(loss3_classifier)
    
    
#    googlenet = Model(input=input, output=[loss1_classifier_act,loss2_classifier_act,loss3_classifier_act])

    
    #lstm1=    TimeDistributed(Dropout(0.5))(lstm1)

 #   lstm2 = LSTM(64 ,return_sequences=True, input_shape=(settings.stepSize,128))(cls5_fc2)
    
   # lstm1 = LSTM(128 ,return_sequences=True, input_shape=(settings.stepSize,512))(lstm)

   # lstm2 = LSTM(64)(lstm1)
     

    cnn_lstm = Model(input=input, output=[pose_xyz,pose_wpqr,cls1_fc_pose_xyz, cls1_fc_pose_wpqr, cls2_fc_pose_xyz, cls2_fc_pose_wpqr])
    
    if weights_path:
        weights_data = np.load(weights_path).item()
        for layer in cnn_lstm.layers:
                if layer.name in weights_data.keys():
                    try:
                        layer_weights = weights_data[layer.name]
                        layer.set_weights((layer_weights['weights'], layer_weights['biases']))
                    except:
                        print('naughty layer!',layer.name)
                else:
                    print('no weights for ',layer.name)
        print("FINISHED SETTING THE WEIGHTS!")
          #cnn_lstm.load_weights(weights_path,by_name=True)
    
    return cnn_lstm



