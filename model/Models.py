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


def get_unet(input_shape, pool_size=(2, 2), n_calsses=1, depth=4, n_base_filters=32):
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = Conv2D(filters=n_base_filters*(2**layer_depth), kernel_size =(2,2), dilation_rate=1 ,padding= 'same',activation= None)(current_layer)
        layer1 = BatchNormalization()(layer1)
        layer1 = Activation(activation='relu')(layer1)
        #print(layer1.shape())
        layer2= Conv2D(filters= n_base_filters*(2**layer_depth)*2, kernel_size =(2,2), dilation_rate=2 ,padding= 'same',activation= None)(layer1)
        layer2 = BatchNormalization()(layer2)
        layer2 = Activation(activation='relu')(layer2)
        #print(layer2.output_shape)

        layer3= Conv2D(filters= n_base_filters*(2**layer_depth)*4, kernel_size =(2,2), dilation_rate=4 ,padding= 'same',activation= None)(layer2)
        layer3 = BatchNormalization()(layer3)
        layer3 = Activation(activation='relu')(layer3)

        layer3= Conv2D(filters= n_base_filters*(2**layer_depth), kernel_size =(1,1), dilation_rate=1 ,padding= 'same',activation= None)(layer3)
        #print(layer3.output_shape)

        # 两个卷积层
        if layer_depth < depth - 1:
            current_layer = MaxPooling2D(pool_size=pool_size)(layer3)
            #print(current_layer.output_shape)
            levels.append([layer1, layer2, layer3, current_layer])
        else:
            current_layer = layer3
            levels.append([layer1, layer2,layer3])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        current_layer = Conv2DTranspose(filters = current_layer._keras_shape[3],
                                        kernel_size = (2,2), strides=(2, 2), padding='same', activation=None)(current_layer)

        up_convolution = Conv2DTranspose(filters = current_layer._keras_shape[3],dilation_rate= 1 ,
                                        kernel_size = (2,2), strides=(1, 1), padding='same', activation=None)(current_layer)
        up_convolution = BatchNormalization()(up_convolution)
        up_convolution = Activation(activation= 'relu')(up_convolution)
        concat = concatenate([up_convolution, levels[layer_depth][2]], axis=3)
        #print(concat.output_shape)

        up_convolution = Conv2DTranspose(filters = current_layer._keras_shape[3]*2,dilation_rate= 2,
                                         kernel_size = (2,2), strides=(1, 1), padding='same', activation=None)(concat)
        up_convolution = BatchNormalization()(up_convolution)
        up_convolution = Activation(activation= 'relu')(up_convolution)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=3)
        #print(concat.output_shape)

        up_convolution = Conv2DTranspose(filters=current_layer._keras_shape[3]*4, dilation_rate= 4,
                                         kernel_size=(2, 2), strides=(1, 1),padding='same', activation=None)(concat)
        up_convolution = BatchNormalization()(up_convolution)
        up_convolution = Activation(activation='relu')(up_convolution)
        concat = concatenate([up_convolution, levels[layer_depth][0]], axis=3)
        #print(concat.output_shape)

        current_layer = Conv2D(kernel_size=(1,1),filters= current_layer._keras_shape[3])(concat)

        #print(current_layer.output_shape)

    current_layer = concatenate([current_layer,inputs],axis =3)
    current_layer0 = Conv2D(kernel_size=(3, 3), padding='same',activation = None,filters = current_layer._keras_shape[3])(current_layer)
    current_layer0 = BatchNormalization()(current_layer0)
    current_layer0 = Activation(activation = 'relu')(current_layer0)
    current_layer0 = concatenate([current_layer,current_layer0],axis =3)

    #print(current_layer.output_shape)

    #SEN = GlobalAveragePooling2D()(current_layer)
    #SEN = MaxoutDense(output_dim= current_layer._keras_shape[3])(SEN)
    #SEN = MaxoutDense(output_dim= current_layer._keras_shape[3]//16)(SEN)
    #SEN = Activation(activation= 'relu')(SEN)
    #current_layer = Multiply()([SEN,current_layer])

    current_layer1 = Conv2D(kernel_size=(2, 2), padding='same',activation = None,filters = n_base_filters)(current_layer0)
    current_layer1 = BatchNormalization()(current_layer1)
    current_layer1 = Activation(activation = 'relu')(current_layer1)
    current_layer1 = concatenate([current_layer1,current_layer0],axis =3)

    #print(current_layer.output_shape)

    current_layer2 = Conv2D(kernel_size=(1, 1), padding='same',activation = None,filters = n_base_filters)(current_layer1)
    current_layer2 = BatchNormalization()(current_layer2)
    current_layer2 = Activation(activation = 'relu')(current_layer2)
    current_layer2 = concatenate([current_layer1,current_layer2],axis =3)

    #print(current_layer.output_shape)

    current_layer = Conv2D(kernel_size=(1, 1), padding='same',activation = None,filters = n_calsses)(current_layer2)
    current_layer = BatchNormalization()(current_layer)
    current_layer = Activation(activation = 'sigmoid')(current_layer)
    #print(current_layer.output_shape)

    model = Model(inputs = inputs,outputs= current_layer)
    return model


def get_unet_FPN(input_shape, pool_size=(2, 2), n_calsses=1, depth=4, n_base_filters=32):
    inputs = Input(input_shape)

    FPNS = []

    levels = list()
    current_layer = inputs
    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = Conv2D(filters=n_base_filters*(1+layer_depth), kernel_size =(3,3), dilation_rate=1 ,padding= 'same',activation= None)(current_layer)
        layer1 = BatchNormalization()(layer1)
        layer1 = Activation(activation='relu')(layer1)
        #print(layer1.shape())
        layer2= Conv2D(filters= n_base_filters*(1+layer_depth), kernel_size =(2,2), dilation_rate=2 ,padding= 'same',activation= None)(layer1)
        layer2 = BatchNormalization()(layer2)
        layer2 = Activation(activation='relu')(layer2)
        #print(layer2.output_shape)

        layer3= Conv2D(filters= n_base_filters*(1+layer_depth), kernel_size =(1,1), dilation_rate=4 ,padding= 'same',activation= None)(layer2)
        layer3 = BatchNormalization()(layer3)
        layer3 = Activation(activation='relu')(layer3)

        layer3= Conv2D(filters= n_base_filters*(1+layer_depth), kernel_size =(1,1), dilation_rate=1 ,padding= 'same',activation= None)(layer3)
        #print(layer3.output_shape)

        # 两个卷积层
        if layer_depth < depth - 1:
            current_layer = MaxPooling2D(pool_size=pool_size)(layer3)
            #print(current_layer.output_shape)
            levels.append([layer1, layer2, layer3, current_layer])
        else:
            current_layer = layer3
            levels.append([layer1, layer2,layer3])

    fpn = layer3
    for temp in range(depth - 1):
        fpn =   Conv2DTranspose(filters = current_layer._keras_shape[3],dilation_rate= 1 ,
                                    kernel_size = (2,2), strides=(2,2), padding='same', activation=None)(fpn)

    FPNS += [fpn]

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        current_layer = Conv2DTranspose(filters = n_base_filters*(1+layer_depth), dilation_rate= 1 ,
                                    kernel_size = (2,2), strides=(2,2), padding='same', activation=None)(current_layer)

        up_convolution = Conv2DTranspose(filters = current_layer._keras_shape[3],dilation_rate= 1 ,
                                        kernel_size = (1,1), strides=(1, 1), padding='same', activation=None)(current_layer)
        up_convolution = Activation(activation= 'relu')(up_convolution)
        concat = concatenate([up_convolution, levels[layer_depth][2]], axis=3)
        #print(concat.output_shape)

        up_convolution = Conv2DTranspose(filters = n_base_filters*(1+layer_depth),dilation_rate= 2,
                                         kernel_size = (2,2), strides=(1, 1), padding='same', activation=None)(concat)
        up_convolution = Activation(activation= 'relu')(up_convolution)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=3)

        #print(concat.output_shape)

        up_convolution = Conv2DTranspose(filters=n_base_filters*(1+layer_depth), dilation_rate= 4,
                                         kernel_size=(3, 3), strides=(1, 1),padding='same', activation=None)(concat)
        up_convolution = Activation(activation='relu')(up_convolution)
        concat = concatenate([up_convolution, levels[layer_depth][0]], axis=3)
        
        fpn = concat
        for temp in range(layer_depth):
            fpn =   Conv2DTranspose(filters = n_base_filters*(1+layer_depth),dilation_rate= 1 ,
                                        kernel_size = (2,2), strides=(2,2), padding='same', activation=None)(fpn)
        FPNS += [fpn]
        #print(concat.output_shape)

        current_layer = Conv2D(kernel_size=(1,1),filters= current_layer._keras_shape[3])(concat)

        #print(current_layer.output_shape)

    for depth_up in range(len(FPNS)-1):
        feature_layer = FPNS[depth_up]
        for temp in range(depth-1-depth_up):
            feature_layer = Conv2DTranspose(filters = n_base_filters*(1+layer_depth), dilation_rate= 1 ,
                                    kernel_size = (2,2), strides=(2,2), padding='same', activation=None)(feature_layer)
        if depth_up != 0:
            feature_layer = concatenate([current_layer, feature_layer], axis=3)
            fpn = current_layer
            for temp in range(depth-depth_up-1):
                fpn =   MaxPooling2D(pool_size=pool_size)(fpn)
            FPNS += [fpn]

        feature_layer = Conv2D(filters=n_base_filters*(depth-depth_up), kernel_size =(3,3), dilation_rate=1 ,padding= 'same',activation= None)(feature_layer)
        feature_layer = BatchNormalization()(feature_layer)
        feature_layer = Activation(activation='relu')(feature_layer)

        feature_layer = Conv2D(filters=n_base_filters*(depth-depth_up), kernel_size =(3,3), dilation_rate=1 ,padding= 'same',activation= None)(feature_layer)
        feature_layer = BatchNormalization()(feature_layer)
        feature_layer = Activation(activation='relu')(feature_layer)

        current_layer = MaxPooling2D(pool_size=pool_size)(feature_layer)


    FPNS += [current_layer,inputs]

    current_layer0 = concatenate(FPNS,axis = 3)
    current_layer1 = Conv2D(kernel_size=(3, 3), padding='same',activation = None,filters = 4*n_base_filters)(current_layer0)
    current_layer1 = Activation(activation = 'relu')(current_layer1)
    current_layer1 = concatenate([current_layer0,current_layer1],axis = 3)
    #print(current_layer.output_shape)

    current_layer2 = Conv2D(kernel_size=(2, 2), padding='same',activation = None,filters = 2*n_base_filters)(current_layer1)
    current_layer2 = Activation(activation = 'relu')(current_layer2)
    current_layer2 = concatenate([current_layer1,current_layer2],axis = 3)
    #print(current_layer.output_shape)
    current_layer3 = Conv2D(kernel_size=(1, 1), padding='same',activation = None,filters = n_base_filters)(current_layer2)
    current_layer3 = Activation(activation = 'relu')(current_layer3)
    current_layer3 = concatenate([current_layer2,current_layer3],axis = 3)

    current_layer3 = Conv2D(kernel_size=(1, 1), padding='same',activation = None,filters = n_base_filters)(current_layer2)
    current_layer3 = Activation(activation = 'relu')(current_layer3)

    current_layer = Conv2D(kernel_size=(1, 1), padding='same',activation = None,filters = n_calsses)(current_layer3)
    current_layer = Activation(activation = 'sigmoid')(current_layer)
    #print(current_layer.output_shape)

    model = Model(inputs = inputs,outputs= current_layer)
    return model




def SEN_block(current_layer):
    SEN = GlobalAveragePooling2D()(current_layer)
    #SEN = MaxoutDense(output_dim= current_layer._keras_shape[3]//16)(SEN)
    #SEN = MaxoutDense(output_dim= current_layer._keras_shape[3])(SEN)
    SEN = Dense(current_layer._keras_shape[3]//4,activation = 'relu')(SEN)
    SEN = Dense(current_layer._keras_shape[3],activation = 'relu')(SEN)
    SEN = Activation(activation= 'sigmoid')(SEN)
    current_layer = Multiply()([SEN,current_layer])
    return current_layer

    
def get_unet_SEN(input_shape, pool_size=(2, 2), n_calsses=1, depth=4, n_base_filters=32):
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = Conv2D(filters=n_base_filters*(2**layer_depth), kernel_size =(3,3), dilation_rate=1 ,padding= 'same',activation= None)(current_layer)
        #layer1 = BatchNormalization()(layer1)
        layer1 = Activation(activation='relu')(layer1)
        #print(layer1.shape())
        layer1 = SEN_block(layer1)
        
        layer2= Conv2D(filters= n_base_filters*(2**layer_depth)*2, kernel_size =(3,3), dilation_rate=2 ,padding= 'same',activation= None)(layer1)
        #layer2 = BatchNormalization()(layer2)
        layer2 = Activation(activation='relu')(layer2)
        layer2 = SEN_block(layer2)

        #print(layer2.output_shape)

        layer3= Conv2D(filters= n_base_filters*(2**layer_depth)*4, kernel_size =(3,3), dilation_rate=4 ,padding= 'same',activation= None)(layer2)
        layer3 = BatchNormalization()(layer3)
        layer3 = Activation(activation='relu')(layer3)
        layer3= Conv2D(filters= n_base_filters*(2**layer_depth), kernel_size =(1,1), dilation_rate=1 ,padding= 'same',activation= None)(layer3)
        layer3 = SEN_block(layer3)

        #print(layer3.output_shape)

        # 两个卷积层
        if layer_depth < depth - 1:
            current_layer = MaxPooling2D(pool_size=pool_size)(layer3)
            #print(current_layer.output_shape)
            levels.append([layer1, layer2, layer3, current_layer])
        else:
            current_layer = layer3
            levels.append([layer1, layer2,layer3])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        current_layer = UpSampling2D(size=(2,2))(current_layer)

        up_convolution = Conv2DTranspose(filters = current_layer._keras_shape[3],dilation_rate= 1 ,
                                        kernel_size = (2,2), strides=(1, 1), padding='same', activation=None)(current_layer)
        up_convolution = BatchNormalization()(up_convolution)
        up_convolution = Activation(activation= 'relu')(up_convolution)
        concat = concatenate([up_convolution, levels[layer_depth][0]], axis=3)
        concat = SEN_block(concat)

        
        #print(concat.output_shape)

        up_convolution = Conv2DTranspose(filters = current_layer._keras_shape[3]*2,dilation_rate= 2,
                                         kernel_size = (2,2), strides=(1, 1), padding='same', activation=None)(concat)
        up_convolution = BatchNormalization()(up_convolution)
        up_convolution = Activation(activation= 'relu')(up_convolution)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=3)
        concat = SEN_block(concat)
        #print(concat.output_shape)

        up_convolution = Conv2DTranspose(filters=current_layer._keras_shape[3]*4, dilation_rate= 4,
                                         kernel_size=(2, 2), strides=(1, 1),padding='same', activation=None)(concat)
        up_convolution = BatchNormalization()(up_convolution)
        up_convolution = Activation(activation='relu')(up_convolution)
        concat = concatenate([up_convolution, levels[layer_depth][2]], axis=3)
        concat = SEN_block(concat)
        #print(concat.output_shape)

        current_layer = Conv2D(kernel_size=(1,1),filters= current_layer._keras_shape[3])(concat)

        #print(current_layer.output_shape)

    current_layer = Conv2D(kernel_size=(1, 1), padding='same',activation = None,filters = current_layer._keras_shape[3])(current_layer)
    current_layer = BatchNormalization()(current_layer)
    current_layer = Activation(activation = 'relu')(current_layer)
    current_layer = SEN_block(current_layer)

    #print(current_layer.output_shape)



    current_layer = Conv2D(kernel_size=(3, 3), padding='same',activation = None,filters = n_base_filters)(current_layer)
    current_layer = BatchNormalization()(current_layer)
    current_layer = Activation(activation = 'relu')(current_layer)
    current_layer = SEN_block(current_layer)
    #print(current_layer.output_shape)

    current_layer = Conv2D(kernel_size=(3, 3), padding='same',activation = None,filters = n_base_filters)(current_layer)
    current_layer = BatchNormalization()(current_layer)
    current_layer = Activation(activation = 'relu')(current_layer)
    current_layer = SEN_block(current_layer)
    
    #print(current_layer.output_shape)
    current_layer = Conv2D(kernel_size=(1, 1), padding='same',activation = None,filters = n_calsses)(current_layer)
    current_layer = BatchNormalization()(current_layer)
    current_layer = Activation(activation = 'sigmoid')(current_layer)
    #print(current_layer.output_shape)

    model = Model(inputs = inputs,outputs= current_layer)
    return model


def get_unet_SEN_Dense(input_shape, pool_size=(2, 2), n_calsses=1, depth=4, n_base_filters=32):
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = Conv2D(filters=n_base_filters * (2 ** layer_depth), kernel_size=(3, 3), dilation_rate=1,
                        padding='same', activation=None)(current_layer)
        #layer1 = BatchNormalization()(layer1)
        layer1 = Activation(activation='relu')(layer1)
        layer1 = concatenate([current_layer, layer1], axis=3)
        layer1 = SEN_block(layer1)
        layer1 = Conv2D(kernel_size=(1, 1), padding='same', activation='relu', filters=n_base_filters * (2 ** layer_depth))(layer1)

        layer2 = Conv2D(filters=n_base_filters * (2 ** layer_depth), kernel_size=(3, 3), dilation_rate=2,padding='same', activation=None)(layer1)
        #layer2 = BatchNormalization()(layer2)
        layer2 = Activation(activation='relu')(layer2)
        layer2 = concatenate([current_layer, layer1, layer2], axis=3)
        layer2 = SEN_block(layer2)
        layer2 = Conv2D(kernel_size=(1, 1), padding='same', activation='relu', filters=n_base_filters * (2 ** layer_depth))(layer2)

        # print(layer2.output_shape)

        layer3 = Conv2D(filters=n_base_filters * (2 ** layer_depth), kernel_size=(3, 3), dilation_rate=4,
                        padding='same', activation=None)(layer2)
        #layer3 = BatchNormalization()(layer3)
        layer3 = Activation(activation='relu')(layer3)
        layer3 = Conv2D(filters=n_base_filters * (2 ** layer_depth), kernel_size=(1, 1), dilation_rate=1,
                        padding='same', activation=None)(layer3)
        layer3 = concatenate([current_layer, layer1, layer2, layer3], axis=3)
        layer3 = SEN_block(layer3)
        layer3 = Conv2D(kernel_size=(1, 1), padding='same', activation='relu', filters=n_base_filters * (2 ** layer_depth))(layer3)

        # print(layer3.output_shape)

        # 两个卷积层
        if layer_depth < depth - 1:
            current_layer = MaxPooling2D(pool_size=pool_size)(layer3)
            # print(current_layer.output_shape)
            levels.append([layer1, layer2, layer3, current_layer])
        else:
            current_layer = layer3
            levels.append([layer1, layer2, layer3])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth - 2, -1, -1):
        current_layer = UpSampling2D(size=(2, 2))(current_layer)
        up_convolution = Conv2DTranspose(filters=current_layer._keras_shape[3], dilation_rate=1,
                                         kernel_size=(2, 2), strides=(1, 1), padding='same', activation=None)(current_layer)
        #up_convolution = BatchNormalization()(up_convolution)
        up_convolution = Activation(activation='relu')(up_convolution)
        concat1 = concatenate([up_convolution, levels[layer_depth][0]], axis=3)
        concat1 = concatenate([current_layer, concat1], axis=3)
        concat1 = SEN_block(concat1)
        concat1 = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                        filters=n_base_filters * (2 ** layer_depth))(concat1)

        # print(concat.output_shape)

        up_convolution = Conv2DTranspose(filters=current_layer._keras_shape[3], dilation_rate=2,
                                         kernel_size=(2, 2), strides=(1, 1), padding='same', activation=None)(concat1)
        #up_convolution = BatchNormalization()(up_convolution)
        up_convolution = Activation(activation='relu')(up_convolution)
        concat2 = concatenate([up_convolution, levels[layer_depth][1]], axis=3)
        concat2 = concatenate([current_layer, concat1,concat2], axis=3)
        concat2 = SEN_block(concat2)
        concat2 = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                        filters=current_layer._keras_shape[3])(concat2)

        # print(concat.output_shape)

        up_convolution = Conv2DTranspose(filters=current_layer._keras_shape[3], dilation_rate=4,
                                         kernel_size=(2, 2), strides=(1, 1), padding='same', activation=None)(concat2)
        #up_convolution = BatchNormalization()(up_convolution)
        up_convolution = Activation(activation='relu')(up_convolution)
        concat3 = concatenate([up_convolution, levels[layer_depth][2]], axis=3)
        concat3 = concatenate([current_layer, concat1, concat2, concat3], axis=3)
        concat3 = SEN_block(concat3)
        # print(concat.output_shape)
        concat3 = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                        filters=current_layer._keras_shape[3])(concat3)

        current_layer = Conv2D(kernel_size=(1, 1), filters=current_layer._keras_shape[3])(concat3)
        # print(current_layer.output_shape)

    current_layer1 = Conv2D(kernel_size=(1, 1), padding='same', activation=None, filters=current_layer._keras_shape[3])(
        current_layer)
    #current_layer1 = BatchNormalization()(current_layer1)
    current_layer1 = Activation(activation='relu')(current_layer1)
    current_layer1 = concatenate([current_layer, current_layer1], axis=3)
    current_layer1 = SEN_block(current_layer1)
    current_layer1 = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                     filters=current_layer._keras_shape[3])(current_layer1)
    # print(current_layer.output_shape)

    current_layer2 = Conv2D(kernel_size=(3, 3), padding='same', activation=None, filters=n_base_filters)(current_layer1)
    #current_layer2 = BatchNormalization()(current_layer2)
    current_layer2 = Activation(activation='relu')(current_layer2)
    current_layer2 = concatenate([current_layer, current_layer1,current_layer2], axis=3)
    current_layer2 = SEN_block(current_layer2)
    current_layer2 = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                     filters=n_base_filters)(current_layer2)
    # print(current_layer.output_shape)

    current_layer3 = Conv2D(kernel_size=(2, 2), padding='same', activation=None, filters=n_base_filters)(current_layer2)
    #current_layer3 = BatchNormalization()(current_layer3)
    current_layer3 = Activation(activation='relu')(current_layer3)
    current_layer3 = concatenate([current_layer, current_layer1,current_layer2,current_layer3], axis=3)
    current_layer3 = SEN_block(current_layer3)
    current_layer3 = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                     filters=n_base_filters)(current_layer3)
    # print(current_layer.output_shape)
    current_layer = Conv2D(kernel_size=(1, 1), padding='same', activation=None, filters=n_calsses)(current_layer3)
    #current_layer = BatchNormalization()(current_layer)
    current_layer = Activation(activation='sigmoid')(current_layer)
    # print(current_layer.output_shape)

    model = Model(inputs=inputs, outputs=current_layer)
    return model

def get_unet_SEN_Dense_Res(input_shape, pool_size=(2, 2), n_calsses=1, depth=4, n_base_filters=32):
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        current_layer = Conv2D(kernel_size=(1, 1), padding='same', activation='relu', filters=n_base_filters * (2 ** layer_depth))(current_layer)

        layer1 = Conv2D(filters=n_base_filters * (2 ** layer_depth), kernel_size=(3, 3), dilation_rate=1,
                        padding='same', activation=None)(current_layer)
        #layer1 = BatchNormalization()(layer1)
        layer1 = Activation(activation='relu')(layer1)
        layer1 = concatenate([current_layer, layer1], axis=3)
        layer1 = SEN_block(layer1)
        layer1 = Conv2D(kernel_size=(1, 1), padding='same', activation=None, filters=n_base_filters * (2 ** layer_depth))(layer1)

        layer2 = Conv2D(filters=n_base_filters * (2 ** layer_depth), kernel_size=(3, 3), dilation_rate=2,padding='same', activation=None)(layer1)
        #layer2 = BatchNormalization()(layer2)
        layer2 = Activation(activation='relu')(layer2)

        layer2 = concatenate([current_layer, layer1, layer2], axis=3)
        layer2 = SEN_block(layer2)
        layer2 = Conv2D(kernel_size=(1, 1), padding='same', activation=None, filters=n_base_filters * (2 ** layer_depth))(layer2)
        # print(layer2.output_shape)
        layer2 = Add()([layer2,current_layer])
        
        layer3 = Conv2D(filters=n_base_filters * (2 ** layer_depth), kernel_size=(3, 3), dilation_rate=4,
                        padding='same', activation=None)(layer2)
        #layer3 = BatchNormalization()(layer3)
        layer3 = Activation(activation='relu')(layer3)
        layer3 = Conv2D(filters=n_base_filters * (2 ** layer_depth), kernel_size=(1, 1), dilation_rate=1,
                        padding='same', activation=None)(layer3)
        layer3 = concatenate([current_layer, layer1, layer2, layer3], axis=3)
        layer3 = SEN_block(layer3)
        layer3 = Conv2D(kernel_size=(1, 1), padding='same', activation=None, filters=n_base_filters * (2 ** layer_depth))(layer3)
        layer3 = Add()([layer1,layer3])

        # 单一的残差块不能起作用,感觉我这种方式加入残差模块也不会起作用的样子
        # print(layer3.output_shape)

        # 两个卷积层
        if layer_depth < depth - 1:
            current_layer = MaxPooling2D(pool_size=pool_size)(layer3)
            # print(current_layer.output_shape)
            levels.append([layer1, layer2, layer3, current_layer])
        else:
            current_layer = layer3
            levels.append([layer1, layer2, layer3])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth - 2, -1, -1):
        current_layer = UpSampling2D(size=(2, 2))(current_layer)
        current_layer = Conv2D(kernel_size=(1, 1), filters=n_base_filters * (2 ** layer_depth),activation='relu')(current_layer)

        up_convolution = Conv2DTranspose(filters=n_base_filters * (2 ** layer_depth), dilation_rate=1,
                                         kernel_size=(2, 2), strides=(1, 1), padding='same', activation=None)(current_layer)
        #up_convolution = BatchNormalization()(up_convolution)
        up_convolution = Activation(activation='relu')(up_convolution)
        concat1 = concatenate([up_convolution, levels[layer_depth][0]], axis=3)
        concat1 = concatenate([current_layer, concat1], axis=3)
        concat1 = SEN_block(concat1)
        concat1 = Conv2D(kernel_size=(1, 1), padding='same', activation=None,
                        filters=n_base_filters * (2 ** layer_depth))(concat1)

        # print(concat.output_shape)

        up_convolution = Conv2DTranspose(filters=n_base_filters * (2 ** layer_depth), dilation_rate=2,
                                         kernel_size=(2, 2), strides=(1, 1), padding='same', activation=None)(concat1)
        #up_convolution = BatchNormalization()(up_convolution)
        up_convolution = Activation(activation='relu')(up_convolution)
        concat2 = concatenate([up_convolution, levels[layer_depth][1]], axis=3)
        concat2 = concatenate([current_layer, concat1,concat2], axis=3)
        concat2 = SEN_block(concat2)
        concat2 = Conv2D(kernel_size=(1, 1), padding='same', activation=None,
                        filters=current_layer._keras_shape[3])(concat2)
        concat2 = Add()([concat2,current_layer])

        # print(concat.output_shape)

        up_convolution = Conv2DTranspose(filters=n_base_filters * (2 ** layer_depth), dilation_rate=4,
                                         kernel_size=(2, 2), strides=(1, 1), padding='same', activation=None)(concat2)
        #up_convolution = BatchNormalization()(up_convolution)
        up_convolution = Activation(activation='relu')(up_convolution)
        concat3 = concatenate([up_convolution, levels[layer_depth][2]], axis=3)
        concat3 = concatenate([current_layer, concat1, concat2, concat3], axis=3)
        concat3 = SEN_block(concat3)
        # print(concat.output_shape)
        concat3 = Conv2D(kernel_size=(1, 1), padding='same', activation=None,
                        filters=n_base_filters * (2 ** layer_depth))(concat3)

        current_layer = Add()([concat3,concat1])

        # print(current_layer.output_shape)

    current_layer1 = Conv2D(kernel_size=(3, 3), padding='same', activation=None, filters=current_layer._keras_shape[3])(
        current_layer)
    #current_layer1 = BatchNormalization()(current_layer1)
    current_layer1 = Activation(activation='relu')(current_layer1)
    #current_layer1 = concatenate([current_layer, current_layer1], axis=3)
    #current_layer1 = SEN_block(current_layer1)
    current_layer1 = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                     filters=current_layer._keras_shape[3])(current_layer1)
    # print(current_layer.output_shape)

    current_layer2 = Conv2D(kernel_size=(2, 2), padding='same', activation=None, filters=n_base_filters)(current_layer1)
    #current_layer2 = BatchNormalization()(current_layer2)
    current_layer2 = Activation(activation='relu')(current_layer2)
    #current_layer2 = concatenate([current_layer, current_layer1,current_layer2], axis=3)
    #current_layer2 = SEN_block(current_layer2)
    current_layer2 = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                     filters=n_base_filters)(current_layer2)
    # print(current_layer.output_shape)

    current_layer3 = Conv2D(kernel_size=(1, 1), padding='same', activation=None, filters=n_base_filters)(current_layer2)
    #current_layer3 = BatchNormalization()(current_layer3)
    current_layer3 = Activation(activation='relu')(current_layer3)
    #current_layer3 = concatenate([current_layer, current_layer1,current_layer2,current_layer3], axis=3)
    #current_layer3 = SEN_block(current_layer3)

    current_layer3 = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                     filters=n_base_filters)(current_layer3)
    # print(current_layer.output_shape)
    current_layer = Conv2D(kernel_size=(1, 1), padding='same', activation=None, filters=n_calsses)(current_layer3)
    #current_layer = BatchNormalization()(current_layer)
    current_layer = Activation(activation='sigmoid')(current_layer)
    # print(current_layer.output_shape)

    model = Model(inputs=inputs, outputs=current_layer)
    return model


def get_unet_SEN_Dense_Res_Deconv(input_shape, pool_size=(2, 2), n_calsses=1, depth=4, n_base_filters=32):
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        current_layer = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                               filters=n_base_filters * (2 ** layer_depth))(current_layer)

        layer1 = Conv2D(filters=n_base_filters * (2 ** layer_depth), kernel_size=(3, 3), dilation_rate=1,
                        padding='same', activation=None)(current_layer)
        layer1 = BatchNormalization()(layer1)
        layer1 = Activation(activation='relu')(layer1)
        layer1 = concatenate([current_layer, layer1], axis=3)
        layer1 = SEN_block(layer1)
        layer1 = Conv2D(kernel_size=(1, 1), padding='same', activation=None,
                        filters=n_base_filters * (2 ** layer_depth))(layer1)

        layer2 = Conv2D(filters=n_base_filters * (2 ** layer_depth), kernel_size=(3, 3), dilation_rate=2,
                        padding='same', activation=None)(layer1)
        layer2 = BatchNormalization()(layer2)
        layer2 = Activation(activation='relu')(layer2)

        layer2 = concatenate([current_layer, layer1, layer2], axis=3)
        layer2 = SEN_block(layer2)
        layer2 = Conv2D(kernel_size=(1, 1), padding='same', activation=None,
                        filters=n_base_filters * (2 ** layer_depth))(layer2)
        # print(layer2.output_shape)
        layer2 = Add()([layer2, current_layer])

        layer3 = Conv2D(filters=n_base_filters * (2 ** layer_depth), kernel_size=(3, 3), dilation_rate=4,
                        padding='same', activation=None)(layer2)
        layer3 = BatchNormalization()(layer3)
        layer3 = Activation(activation='relu')(layer3)
        layer3 = Conv2D(filters=n_base_filters * (2 ** layer_depth), kernel_size=(1, 1), dilation_rate=1,
                        padding='same', activation=None)(layer3)
        layer3 = concatenate([current_layer, layer1, layer2, layer3], axis=3)
        layer3 = SEN_block(layer3)
        layer3 = Conv2D(kernel_size=(1, 1), padding='same', activation=None,
                        filters=n_base_filters * (2 ** layer_depth))(layer3)
        layer3 = Add()([layer1, layer3])

        # 单一的残差块不能起作用,感觉我这种方式加入残差模块也不会起作用的样子
        # print(layer3.output_shape)

        # 两个卷积层
        if layer_depth < depth - 1:
            current_layer = MaxPooling2D(pool_size=pool_size)(layer3)
            # print(current_layer.output_shape)
            levels.append([layer1, layer2, layer3, current_layer])
        else:
            current_layer = layer3
            levels.append([layer1, layer2, layer3])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth - 2, -1, -1):
        #current_layer = UpSampling2D(size=(2, 2))(current_layer)
        current_layer = Conv2DTranspose(n_base_filters * (2 ** layer_depth),size=(2, 2),strides=(2,2))(current_layer)

        current_layer = Conv2D(kernel_size=(1, 1), filters=n_base_filters * (2 ** layer_depth), activation='relu')(
            current_layer)

        up_convolution = Conv2DTranspose(filters=n_base_filters * (2 ** layer_depth), dilation_rate=1,
                                         kernel_size=(2, 2), strides=(1, 1), padding='same', activation=None)(
            current_layer)
        up_convolution = BatchNormalization()(up_convolution)
        up_convolution = Activation(activation='relu')(up_convolution)
        concat1 = concatenate([up_convolution, levels[layer_depth][0]], axis=3)
        concat1 = concatenate([current_layer, concat1], axis=3)
        concat1 = SEN_block(concat1)
        concat1 = Conv2D(kernel_size=(1, 1), padding='same', activation=None,
                         filters=n_base_filters * (2 ** layer_depth))(concat1)

        # print(concat.output_shape)

        up_convolution = Conv2DTranspose(filters=n_base_filters * (2 ** layer_depth), dilation_rate=2,
                                         kernel_size=(2, 2), strides=(1, 1), padding='same', activation=None)(concat1)
        up_convolution = BatchNormalization()(up_convolution)
        up_convolution = Activation(activation='relu')(up_convolution)
        concat2 = concatenate([up_convolution, levels[layer_depth][1]], axis=3)
        concat2 = concatenate([current_layer, concat1, concat2], axis=3)
        concat2 = SEN_block(concat2)
        concat2 = Conv2D(kernel_size=(1, 1), padding='same', activation=None,
                         filters=current_layer._keras_shape[3])(concat2)
        concat2 = Add()([concat2, current_layer])

        # print(concat.output_shape)

        up_convolution = Conv2DTranspose(filters=n_base_filters * (2 ** layer_depth), dilation_rate=4,
                                         kernel_size=(2, 2), strides=(1, 1), padding='same', activation=None)(concat2)
        up_convolution = BatchNormalization()(up_convolution)
        up_convolution = Activation(activation='relu')(up_convolution)
        concat3 = concatenate([up_convolution, levels[layer_depth][2]], axis=3)
        concat3 = concatenate([current_layer, concat1, concat2, concat3], axis=3)
        concat3 = SEN_block(concat3)
        # print(concat.output_shape)
        concat3 = Conv2D(kernel_size=(1, 1), padding='same', activation=None,
                         filters=n_base_filters * (2 ** layer_depth))(concat3)

        current_layer = Add()([concat3, concat1])

        # print(current_layer.output_shape)

    current_layer1 = Conv2D(kernel_size=(3, 3), padding='same', activation=None, filters=current_layer._keras_shape[3])(
        current_layer)
    # current_layer1 = BatchNormalization()(current_layer1)
    current_layer1 = Activation(activation='relu')(current_layer1)
    # current_layer1 = concatenate([current_layer, current_layer1], axis=3)
    # current_layer1 = SEN_block(current_layer1)
    current_layer1 = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                            filters=current_layer._keras_shape[3])(current_layer1)
    # print(current_layer.output_shape)

    current_layer2 = Conv2D(kernel_size=(2, 2), padding='same', activation=None, filters=n_base_filters)(current_layer1)
    # current_layer2 = BatchNormalization()(current_layer2)
    current_layer2 = Activation(activation='relu')(current_layer2)
    # current_layer2 = concatenate([current_layer, current_layer1,current_layer2], axis=3)
    # current_layer2 = SEN_block(current_layer2)
    current_layer2 = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                            filters=n_base_filters)(current_layer2)
    # print(current_layer.output_shape)

    current_layer3 = Conv2D(kernel_size=(1, 1), padding='same', activation=None, filters=n_base_filters)(current_layer2)
    # current_layer3 = BatchNormalization()(current_layer3)
    current_layer3 = Activation(activation='relu')(current_layer3)
    # current_layer3 = concatenate([current_layer, current_layer1,current_layer2,current_layer3], axis=3)
    # current_layer3 = SEN_block(current_layer3)

    current_layer3 = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                            filters=n_base_filters)(current_layer3)
    # print(current_layer.output_shape)
    current_layer = Conv2D(kernel_size=(1, 1), padding='same', activation=None, filters=n_calsses)(current_layer3)
    # current_layer = BatchNormalization()(current_layer)
    current_layer = Activation(activation='sigmoid')(current_layer)
    # print(current_layer.output_shape)

    model = Model(inputs=inputs, outputs=current_layer)
    return model
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



def get_unet_inception_resX_sen_dense(input_shape, pool_size=(2, 2), n_calsses=1, depth=4, n_base_filters=32):
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        current_layer = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                               filters=n_base_filters * (2 ** layer_depth))(current_layer)

        layer1 = inception_resX_sen_conv_block(current_layer)

        layer1 = BatchNormalization()(layer1)
        layer1 = Activation(activation='relu')(layer1)
        layer1 = concatenate([current_layer, layer1], axis=3)
        layer1 = SEN_block(layer1)
        layer1 = Conv2D(kernel_size=(1, 1), padding='same', activation=None,
                        filters=n_base_filters * (2 ** layer_depth))(layer1)

        layer2 = inception_resX_sen_conv_block(layer1)

        layer2 = BatchNormalization()(layer2)
        layer2 = Activation(activation='relu')(layer2)

        layer2 = concatenate([current_layer, layer1, layer2], axis=3)
        layer2 = SEN_block(layer2)
        layer2 = Conv2D(kernel_size=(1, 1), padding='same', activation=None,
                        filters=n_base_filters * (2 ** layer_depth))(layer2)
        # print(layer2.output_shape)
        layer2 = Add()([layer2, current_layer])

        layer3 = inception_resX_sen_conv_block(layer2)
        layer3 = BatchNormalization()(layer3)
        layer3 = Activation(activation='relu')(layer3)
        layer3 = Conv2D(filters=n_base_filters * (2 ** layer_depth), kernel_size=(1, 1), dilation_rate=1,
                        padding='same', activation=None)(layer3)
        layer3 = concatenate([current_layer, layer1, layer2, layer3], axis=3)
        layer3 = SEN_block(layer3)
        layer3 = Conv2D(kernel_size=(1, 1), padding='same', activation=None,
                        filters=n_base_filters * (2 ** layer_depth))(layer3)
        layer3 = Add()([layer1, layer3])

        # 单一的残差块不能起作用,感觉我这种方式加入残差模块也不会起作用的样子
        # print(layer3.output_shape)

        # 两个卷积层
        if layer_depth < depth - 1:
            current_layer = MaxPooling2D(pool_size=pool_size)(layer3)
            # print(current_layer.output_shape)
            levels.append([layer1, layer2, layer3, current_layer])
        else:
            current_layer = layer3
            levels.append([layer1, layer2, layer3])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth - 2, -1, -1):
        #current_layer = UpSampling2D(size=(2, 2))(current_layer)
        current_layer = Conv2DTranspose(n_base_filters * (2 ** layer_depth),kernel_size=(2, 2),strides=(2,2))(current_layer)
        current_layer = Conv2D(kernel_size=(1, 1), filters=n_base_filters * (2 ** layer_depth), activation='relu')(current_layer)

        up_convolution = inception_resX_sen_conv_block(current_layer)
        up_convolution = BatchNormalization()(up_convolution)
        up_convolution = Activation(activation='relu')(up_convolution)
        concat1 = concatenate([up_convolution, levels[layer_depth][0]], axis=3)
        concat1 = concatenate([current_layer, concat1], axis=3)
        concat1 = SEN_block(concat1)
        concat1 = Conv2D(kernel_size=(1, 1), padding='same', activation=None,
                         filters=n_base_filters * (2 ** layer_depth))(concat1)

        # print(concat.output_shape)

        up_convolution = inception_resX_sen_conv_block(concat1)
        up_convolution = BatchNormalization()(up_convolution)
        up_convolution = Activation(activation='relu')(up_convolution)
        concat2 = concatenate([up_convolution, levels[layer_depth][1]], axis=3)
        concat2 = concatenate([current_layer, concat1, concat2], axis=3)
        concat2 = SEN_block(concat2)
        concat2 = Conv2D(kernel_size=(1, 1), padding='same', activation=None,
                         filters=current_layer._keras_shape[3])(concat2)
        concat2 = Add()([concat2, current_layer])

        # print(concat.output_shape)

        up_convolution = inception_resX_sen_conv_block(concat2)
        up_convolution = BatchNormalization()(up_convolution)
        up_convolution = Activation(activation='relu')(up_convolution)
        concat3 = concatenate([up_convolution, levels[layer_depth][2]], axis=3)
        concat3 = concatenate([current_layer, concat1, concat2, concat3], axis=3)
        concat3 = SEN_block(concat3)
        # print(concat.output_shape)
        concat3 = Conv2D(kernel_size=(1, 1), padding='same', activation=None,
                         filters=n_base_filters * (2 ** layer_depth))(concat3)

        current_layer = Add()([concat3, concat1])

        # print(current_layer.output_shape)

    current_layer1 = inception_resX_sen_conv_block(current_layer)
    #current_layer1 = Activation(activation='relu')(current_layer1)
    current_layer1 = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                            filters=current_layer._keras_shape[3])(current_layer1)

    current_layer2 = inception_resX_sen_conv_block(current_layer1)
    #current_layer2 = Activation(activation='relu')(current_layer2)
    current_layer2 = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                            filters=n_base_filters)(current_layer2)

    current_layer3 = inception_resX_sen_conv_block(current_layer2)
    current_layer3 = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                            filters=n_base_filters)(current_layer3)
    current_layer = Conv2D(kernel_size=(1, 1), padding='same', activation=None, filters=n_calsses)(current_layer3)
    current_layer = Activation(activation='sigmoid')(current_layer)
    model = Model(inputs=inputs, outputs=current_layer)
    return model




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

def get_unet_inception_res_sen_dense(input_shape, pool_size=(2, 2), n_calsses=1, depth=4, n_base_filters=32):
    inputs = Input(input_shape)
    #inputs = Input((None,None,3))
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        current_layer = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                               filters=n_base_filters * (1 + layer_depth))(current_layer)

        layer1 = inception_res_sen_conv_block(current_layer)

        layer1 = BatchNormalization()(layer1)
        layer1 = Activation(activation='relu')(layer1)
        layer1 = concatenate([current_layer, layer1], axis=3)
        layer1 = SEN_block(layer1)
        layer1 = Conv2D(kernel_size=(1, 1), padding='same', activation=None,
                        filters=n_base_filters * (1 + layer_depth))(layer1)

        layer2 = inception_res_sen_conv_block(layer1)

        layer2 = BatchNormalization()(layer2)
        layer2 = Activation(activation='relu')(layer2)

        layer2 = concatenate([current_layer, layer1, layer2], axis=3)
        layer2 = SEN_block(layer2)
        layer2 = Conv2D(kernel_size=(1, 1), padding='same', activation=None,
                        filters=n_base_filters * (1 + layer_depth))(layer2)
        # print(layer2.output_shape)
        layer2 = Add()([layer2, current_layer])

        layer3 = inception_res_sen_conv_block(layer2)
        layer3 = BatchNormalization()(layer3)
        layer3 = Activation(activation='relu')(layer3)
        layer3 = Conv2D(filters=n_base_filters * (1 + layer_depth), kernel_size=(1, 1), dilation_rate=1,
                        padding='same', activation=None)(layer3)
        layer3 = concatenate([current_layer, layer1, layer2, layer3], axis=3)
        layer3 = SEN_block(layer3)
        layer3 = Conv2D(kernel_size=(1, 1), padding='same', activation=None,
                        filters=n_base_filters * (1 + layer_depth))(layer3)
        layer3 = Add()([layer1, layer3])

        # 单一的残差块不能起作用,感觉我这种方式加入残差模块也不会起作用的样子
        # print(layer3.output_shape)

        # 两个卷积层
        if layer_depth < depth - 1:
            current_layer = MaxPooling2D(pool_size=pool_size)(layer3)
            # print(current_layer.output_shape)
            levels.append([layer1, layer2, layer3, current_layer])
        else:
            current_layer = layer3
            levels.append([layer1, layer2, layer3])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth - 2, -1, -1):
        #current_layer = UpSampling2D(size=(2, 2))(current_layer)
        current_layer = Conv2DTranspose(n_base_filters * (1 + layer_depth),kernel_size=(2, 2),strides=(2,2))(current_layer)
        current_layer = Conv2D(kernel_size=(1, 1), filters=n_base_filters * (1 + layer_depth), activation='relu')(current_layer)

        up_convolution = inception_res_sen_conv_block(current_layer)
        up_convolution = BatchNormalization()(up_convolution)
        up_convolution = Activation(activation='relu')(up_convolution)
        concat1 = concatenate([up_convolution, levels[layer_depth][0]], axis=3)
        concat1 = concatenate([current_layer, concat1], axis=3)
        concat1 = SEN_block(concat1)
        concat1 = Conv2D(kernel_size=(1, 1), padding='same', activation=None,
                         filters=n_base_filters * (1 + layer_depth))(concat1)

        # print(concat.output_shape)

        up_convolution = inception_res_sen_conv_block(concat1)
        up_convolution = BatchNormalization()(up_convolution)
        up_convolution = Activation(activation='relu')(up_convolution)
        concat2 = concatenate([up_convolution, levels[layer_depth][1]], axis=3)
        concat2 = concatenate([current_layer, concat1, concat2], axis=3)
        concat2 = SEN_block(concat2)
        concat2 = Conv2D(kernel_size=(1, 1), padding='same', activation=None,
                         filters=current_layer._keras_shape[3])(concat2)
        concat2 = Add()([concat2, current_layer])

        # print(concat.output_shape)

        up_convolution = inception_res_sen_conv_block(concat2)
        up_convolution = BatchNormalization()(up_convolution)
        up_convolution = Activation(activation='relu')(up_convolution)
        concat3 = concatenate([up_convolution, levels[layer_depth][2]], axis=3)
        concat3 = concatenate([current_layer, concat1, concat2, concat3], axis=3)
        concat3 = SEN_block(concat3)
        # print(concat.output_shape)
        concat3 = Conv2D(kernel_size=(1, 1), padding='same', activation=None,
                         filters=n_base_filters * (2 ** layer_depth))(concat3)

        current_layer = Add()([concat3, concat1])

        # print(current_layer.output_shape)

    current_layer1 = inception_res_sen_conv_block(current_layer)
    # current_layer1 = BatchNormalization()(current_layer1)
    current_layer1 = Activation(activation='relu')(current_layer1)
    # current_layer1 = concatenate([current_layer, current_layer1], axis=3)
    # current_layer1 = SEN_block(current_layer1)
    current_layer1 = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                            filters=current_layer._keras_shape[3])(current_layer1)
    # print(current_layer.output_shape)

    current_layer2 = inception_res_sen_conv_block(current_layer1)
    # current_layer2 = BatchNormalization()(current_layer2)
    current_layer2 = Activation(activation='relu')(current_layer2)
    # current_layer2 = concatenate([current_layer, current_layer1,current_layer2], axis=3)
    # current_layer2 = SEN_block(current_layer2)
    current_layer2 = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                            filters=n_base_filters)(current_layer2)
    # print(current_layer.output_shape)

    current_layer3 = Conv2D(kernel_size=(1, 1), padding='same', activation=None, filters=n_base_filters)(current_layer2)
    # current_layer3 = BatchNormalization()(current_layer3)
    current_layer3 = Activation(activation='relu')(current_layer3)
    # current_layer3 = concatenate([current_layer, current_layer1,current_layer2,current_layer3], axis=3)
    # current_layer3 = SEN_block(current_layer3)

    current_layer3 = Conv2D(kernel_size=(1, 1), padding='same', activation='relu',
                            filters=n_base_filters)(current_layer3)
    # print(current_layer.output_shape)
    current_layer = Conv2D(kernel_size=(1, 1), padding='same', activation=None, filters=n_calsses)(current_layer3)
    # current_layer = BatchNormalization()(current_layer)
    current_layer = Activation(activation='sigmoid')(current_layer)
    # print(current_layer.output_shape)
    model = Model(inputs=inputs, outputs=current_layer)
    model.compile(optimizer=SGD(lr=0.001), loss=bce_dice_loss, metrics=[dice_coeff])

    return model

