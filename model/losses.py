from keras.losses import binary_crossentropy
import keras.backend as K
import numpy as np
from scipy.spatial.distance import pdist, cdist
import cv2

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)**4
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def weighted_dice_coeff(y_true, y_pred, weight):
    smooth = 1.
    print(weight.shape)
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    return score
def weighted_dice_coeff_hans(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    return score

def weighted_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    if K.int_shape(y_pred)[1] == 128:
        kernel_size = 11
    elif K.int_shape(y_pred)[1] == 256:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 512:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 1024:
        kernel_size = 41
    else:
        raise ValueError('Unexpected image size')
    averaged_mask = K.pool2d(
        y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = 1 - (weighted_dice_coeff(y_true, y_pred, weight))**2
    return loss


def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))

    # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
                                          (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)


def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    if K.int_shape(y_pred)[1] == 128:
        kernel_size = 11
    elif K.int_shape(y_pred)[1] == 256:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 512:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 1024:
        kernel_size = 41
    else:
        raise ValueError('Unexpected image size')
    averaged_mask = K.pool2d(
        y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', pool_mode='avg')

    print('***************' + str(averaged_mask.shape))
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')

    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + (1 - weighted_dice_coeff(y_true, y_pred, weight))
    return loss


def weighted_bce_dice_loss_hans(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    mask_std = cv2.imread('input/mask_std_temp.jpg', cv2.IMREAD_GRAYSCALE)
    mask_std = cv2.resize(mask_std, (y_pred.shape[1], y_pred.shape[1]))
    averaged_mask = np.expand_dims(np.expand_dims(K.cast_to_floatx(np.array(mask_std)),axis = 2),axis = 0)
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += averaged_mask * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = 1 - (weighted_dice_coeff(y_true, y_pred, weight)) ** 2
    return loss

def weighted_bce_dice_loss_hans_1(y_true, y_pred):
    print(y_true.shape)
    print(y_pred.shape)
    y_true = y_true[0,:,:,0]
    y_pred = y_pred[0,:,:,0]
    print(y_true.shape)
    print(y_pred.shape)

    border = np.abs(np.gradient(y_true)[1]) + np.abs(np.gradient(y_true)[0])
    border = np.select([border == 0.5, border != 0.5], [1.0, border])
    XA = []
    XB = []
    for x, y in zip(np.nonzero(border)[0], np.nonzero(border)[1]):
        XB.append((x, y))
    for x in range(0, y_true.shape[0]):
        for y in range(0, y_true.shape[1]):
            XA.append((x, y))

    dist_matrix = cdist(np.array(XA), np.array(XB), metric="euclidean").min(
        axis=1, keepdims=False).reshape(y_true.shape)
    weights = (1.0 / (dist_matrix + 5.0))
    w0 = K.sum(K.ones_like(y_true))
    w1 = K.sum(weights)
    weights *= (w0 / w1)
    loss = 1 - (weighted_dice_coeff(y_true, y_pred, weights)) ** 2
    return loss