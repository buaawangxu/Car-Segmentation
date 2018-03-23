import numpy as np
import configparser
from model.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff,weighted_bce_dice_loss_hans

from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, AveragePooling2D, Dense,UpSampling2D, Reshape, core, Dropout,BatchNormalization,GlobalAveragePooling2D
from keras.optimizers import Adam,RMSprop,SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model
import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU,Conv2DTranspose,Multiply,Add
from keras import layers

from keras.optimizers import Adam
K.set_image_data_format("channels_last")
try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate



def SEN_block(current_layer):
    SEN = GlobalAveragePooling2D()(current_layer)
    #SEN = MaxoutDense(output_dim= current_layer._keras_shape[3]//16)(SEN)
    #SEN = MaxoutDense(output_dim= current_layer._keras_shape[3])(SEN)
    SEN = Dense(current_layer._keras_shape[3],activation = 'relu')(SEN)
    SEN = Dense(current_layer._keras_shape[3],activation = 'relu')(SEN)
    SEN = Activation(activation= 'sigmoid')(SEN)
    current_layer = Multiply()([SEN,current_layer])
    return current_layer


def resnext_block(current_layer,filters = 8,kernel_size=(1,1),group_num = 8, padding='same',activation='relu'):
    groups = []
    filter_num_i = filters // group_num
    for i in range(group_num):
        conv1 = Conv2D(filters=filter_num_i, kernel_size = (1,1), padding=padding, activation = activation)(current_layer)
        conv1 = Conv2D(filters=filter_num_i, kernel_size = kernel_size, padding=padding, activation = activation)(conv1)
        conv1 = Conv2D(filters=filter_num_i, kernel_size = (1,1), padding=padding, activation = activation)(conv1)
        groups.append(conv1)
    out = concatenate(groups,axis = 3)
    out = Conv2D(filters=filters, kernel_size = (1,1), padding=padding, activation = activation)(out)
    return out


def inception_resX_sen_conv_block(current_layer):
    conv1 = resnext_block(current_layer,filters=current_layer._keras_shape[3],kernel_size=(1,1),padding='same',activation='relu')
    conv1 = resnext_block(conv1,filters=current_layer._keras_shape[3],kernel_size=(1,3),padding='same',activation='relu')
    conv1 = resnext_block(conv1,filters=current_layer._keras_shape[3],kernel_size=(3,1),padding='same',activation='relu')

    conv11 = resnext_block(conv1,filters=current_layer._keras_shape[3],kernel_size=(1,3),padding='same',activation='relu')
    conv12 = resnext_block(conv1,filters=current_layer._keras_shape[3],kernel_size=(3,1),padding='same',activation='relu')

    conv2 = resnext_block(current_layer,filters=current_layer._keras_shape[3], kernel_size=(1, 1), padding='same',activation='relu')
    conv21 = resnext_block(conv2,filters=current_layer._keras_shape[3], kernel_size=(1, 3), padding='same',activation='relu')
    conv22 = resnext_block(conv2,filters=current_layer._keras_shape[3], kernel_size=(3, 1), padding='same',activation='relu')

    conv3 = AveragePooling2D( pool_size =(3, 3),strides=(1,1),padding='same')(current_layer)
    conv3 = resnext_block(conv3,filters=current_layer._keras_shape[3], kernel_size=(1, 1), padding='same',activation='relu')

    conv4 = resnext_block(current_layer,filters=current_layer._keras_shape[3],kernel_size=(1,1),padding='same',activation='relu')

    concat = concatenate([conv11,conv12,conv21,conv22,conv3,conv4],axis = 3)
    concat = Conv2D(filters=current_layer._keras_shape[3],kernel_size=(1,1),padding='same',activation='relu')(concat)
    sen = SEN_block(concat)
    out = Add()([sen,current_layer])
    return out




def inception_res_sen_conv_block(current_layer):

    conv1 = Conv2D(filters=current_layer._keras_shape[3],kernel_size=(1,1),padding='same',activation='relu')(current_layer)
    conv1 = Conv2D(filters=current_layer._keras_shape[3],kernel_size=(1,3),padding='same',activation='relu')(conv1)
    conv1 = Conv2D(filters=current_layer._keras_shape[3],kernel_size=(3,1),padding='same',activation='relu')(conv1)

    conv11 = Conv2D(filters=current_layer._keras_shape[3],kernel_size=(1,3),padding='same',activation='relu')(conv1)
    conv12 = Conv2D(filters=current_layer._keras_shape[3],kernel_size=(3,1),padding='same',activation='relu')(conv1)

    conv2 = Conv2D(filters=current_layer._keras_shape[3], kernel_size=(1, 1), padding='same',activation='relu')(current_layer)
    conv21 = Conv2D(filters=current_layer._keras_shape[3], kernel_size=(1, 3), padding='same',activation='relu')(conv2)
    conv22 = Conv2D(filters=current_layer._keras_shape[3], kernel_size=(3, 1), padding='same',activation='relu')(conv2)

    conv3 = AveragePooling2D( pool_size =(3, 3),strides=(1,1),padding='same')(current_layer)
    conv3 = Conv2D(filters=current_layer._keras_shape[3], kernel_size=(1, 1), padding='same',activation='relu')(conv3)

    conv4 = Conv2D(filters=current_layer._keras_shape[3],kernel_size=(1,1),padding='same',activation='relu')(current_layer)

    concat = concatenate([conv11,conv12,conv21,conv22,conv3,conv4],axis = 3)
    concat = Conv2D(filters=current_layer._keras_shape[3],kernel_size=(1,1),padding='same',activation='relu')(concat)
    sen = SEN_block(concat)
    out = Add()([sen,current_layer])
    return out

def get_post_process_inception_res_sen_dense(input_shape, pool_size=(2, 2), n_calsses=1, n_base_filters=32):
    inputs = Input(input_shape)
    #inputs = Input((None,None,3))
    current_layer = inputs

    tops = [current_layer]

    # 向下结构
    current_layer = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                           filters=n_base_filters)(current_layer)
    layer1 = inception_res_sen_conv_block(current_layer)
    layer2 = inception_res_sen_conv_block(layer1)
    layer3 = inception_res_sen_conv_block(layer2)
    layer3 = Add()([layer1, layer3])
    current_layer = MaxPooling2D(pool_size=pool_size)(layer3)
    

    current_layer = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                           filters=n_base_filters)(current_layer)
    layer1 = inception_res_sen_conv_block(current_layer)
    layer2 = inception_res_sen_conv_block(layer1)
    layer3 = inception_res_sen_conv_block(layer2)
    layer3 = Add()([layer1, layer3])

    current_layer = Conv2DTranspose(n_base_filters, kernel_size=(2, 2),strides=(2,2))(layer3)
    current_layer = Conv2D(kernel_size=(1, 1), filters=n_base_filters, activation='relu')(current_layer)

    up_convolution1 = inception_res_sen_conv_block(current_layer)
    up_convolution2 = inception_res_sen_conv_block(up_convolution1)
    up_convolution3 = inception_res_sen_conv_block(up_convolution2)
    current_layer = Add()([up_convolution1, up_convolution3])

    tops.append(current_layer)

    current_layer = inputs
    current_layer = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                           filters=n_base_filters)(current_layer)

    layer1 = inception_res_sen_conv_block(current_layer)
    layer2 = inception_res_sen_conv_block(layer1)
    layer3 = inception_res_sen_conv_block(layer2)
    layer3 = Add()([layer1, layer3])
    current_layer = Conv2DTranspose(n_base_filters, kernel_size=(2, 2),strides=(2,2))(layer3)
    

    current_layer = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                           filters=n_base_filters)(current_layer)
    layer1 = inception_res_sen_conv_block(current_layer)
    layer2 = inception_res_sen_conv_block(layer1)
    layer3 = inception_res_sen_conv_block(layer2)
    layer3 = Add()([layer1, layer3])

    current_layer = MaxPooling2D(pool_size=pool_size)(layer3)
    current_layer = Conv2D(kernel_size=(1, 1), filters=n_base_filters, activation='relu')(current_layer)
    up_convolution1 = inception_res_sen_conv_block(current_layer)
    up_convolution2 = inception_res_sen_conv_block(up_convolution1)
    up_convolution3 = inception_res_sen_conv_block(up_convolution2)
    current_layer = Add()([up_convolution1, up_convolution3])

    tops.append(current_layer)

    top = concatenate(tops,axis = 3)

    # 顶层的DenseNet结构，强化边缘

    current_layer1 = inception_res_sen_conv_block(top)
    concat_top1 = concatenate([top, current_layer1], axis=3)

    current_layer2 = inception_res_sen_conv_block(concat_top1)
    concat_top2 = concatenate([current_layer2, concat_top1], axis=3)
    current_layer3 = inception_res_sen_conv_block(concat_top2)
    concat_top3 = concatenate([current_layer3, concat_top2], axis=3)

    current_layer3 = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                            filters=n_base_filters)(concat_top3)
    current_layer = Conv2D(kernel_size=(1, 1), padding='same', activation=None, filters=n_calsses)(current_layer3)
    current_layer = Activation(activation='sigmoid')(current_layer)
    model = Model(inputs=inputs, outputs=current_layer)
    model.compile(optimizer=SGD(lr=0.001), loss=bce_dice_loss, metrics=[dice_coeff])

    return model

