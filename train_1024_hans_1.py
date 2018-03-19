import cv2
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from model.u_net import get_unet_1024_PReLU_Hans_1,get_unet_1024_PReLU,get_unet_1024_LeakyReLu
from sklearn.utils import shuffle
from keras.optimizers import RMSprop,Adam,sgd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
#set_session(tf.Session(config=config))

import params

input_size = params.input_size
epochs = params.max_epochs
batch_size = params.batch_size
model =get_unet_1024_LeakyReLu((1024,1024,5))
#model = get_unet_1024_PReLU_Hans_1()

df_train = pd.read_csv('input/train_masks.csv')
ids_train = df_train['img'].map(lambda s: s.split('.')[0])

ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=42)

print('Training on {} samples'.format(len(ids_train_split)))
print('Validating on {} samples'.format(len(ids_valid_split)))

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask,mask_mean,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))
        mask_mean = cv2.warpPerspective(mask_mean, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))
    return image, mask, mask_mean

def randomHorizontalFlip(image, mask,mask_mean,u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        mask_mean = cv2.flip(mask_mean, 1)
    return image, mask, mask_mean

def train_generator():
    while True:
        global ids_train_split
        for start in range(0, len(ids_train_split), batch_size):
            ids_train_split = shuffle(ids_train_split)
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]
            for id in ids_train_batch.values:
                print('\nTraining Generating...'+id +'/'+str(len(ids_train_batch.values)))
                img = cv2.imread('input/train/{}.jpg'.format(id))
                img = cv2.resize(img, (input_size, input_size))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #clahe = cv2.createCLAHE()
                #img_gray = clahe.apply(img_gray)
                img_gray = cv2.equalizeHist(img_gray)
                #img[:,:,0] = cv2.equalizeHist(img[:,:,0])
                #img[:,:,1] = cv2.equalizeHist(img[:,:,1])
                #img[:,:,2] = cv2.equalizeHist(img[:,:,2])

                mask = cv2.imread('input/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
                mask_mean = cv2.imread('input/mask_mean_{}.jpg'.format(id[-2:]), cv2.IMREAD_GRAYSCALE)
                mask_mean = cv2.resize(mask_mean, (input_size, input_size))

                mask = cv2.resize(mask, (input_size, input_size))
                img = randomHueSaturationValue(img,
                                              hue_shift_limit=(-50, 50),
                                              sat_shift_limit=(-5, 5),
                                               val_shift_limit=(-15, 15))
                img, mask,img_gray = randomShiftScaleRotate(img, mask, img_gray,
                                                              shift_limit=(-0.0625, 0.0625),
                                                              scale_limit=(-0.1, 0.1),
                                                             rotate_limit=(-0, 0))
                # img, mask, mask_mean = randomHorizontalFlip(img, mask, mask_mean)
                mask_mean = np.expand_dims(mask_mean, axis=2) / 255
                mask = np.expand_dims(mask, axis=2) / 255
                img_gray = np.expand_dims(img_gray, axis=2) / 255
                img = img / 255
                #print(img.shape,mask_mean.shape,img_gray.shape)
                img = np.concatenate([img, mask_mean, img_gray], axis=2)
                #img -= img.mean(axis=(0,1,2),keepdims = True)
                #img /= img.std(axis=(0,1,2),keepdims = True)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            yield x_batch, y_batch

def valid_generator():
    while True:
        global ids_valid_split
        for start in range(0, len(ids_valid_split), batch_size):
            ids_valid_split = shuffle(ids_valid_split)
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_valid_split))
            ids_valid_batch = ids_valid_split[start:end]
            for id in ids_valid_batch.values:
                img = cv2.imread('input/train/{}.jpg'.format(id))
                img = cv2.resize(img, (input_size, input_size))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_gray = cv2.equalizeHist(img_gray)
                #clahe = cv2.createCLAHE()
                #img_gray = clahe.apply(img_gray)


                #img[:,:,0] = cv2.equalizeHist(img[:,:,0])
                #img[:,:,1] = cv2.equalizeHist(img[:,:,1])
                #img[:,:,2] = cv2.equalizeHist(img[:,:,2])
                mask = cv2.imread('input/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
                mask_mean = cv2.imread('input/mask_mean_{}.jpg'.format(id[-2:]), cv2.IMREAD_GRAYSCALE)
                mask_mean = cv2.resize(mask_mean, (input_size, input_size))

                mask = cv2.resize(mask, (input_size, input_size))
                # img = randomHueSaturationValue(img,
                #                              hue_shift_limit=(-50, 50),
                #                              sat_shift_limit=(-5, 5),
                #                               val_shift_limit=(-15, 15))
                # img, mask, mask_mean = randomShiftScaleRotate(img, mask, mask_mean0,
                #                                              shift_limit=(-0.0625, 0.0625),
                #                                              scale_limit=(-0.1, 0.1),
                #                                              rotate_limit=(-0, 0))
                # img, mask, mask_mean = randomHorizontalFlip(img, mask, mask_mean)
                mask_mean = np.expand_dims(mask_mean, axis=2) / 255
                mask = np.expand_dims(mask, axis=2) / 255
                img_gray = np.expand_dims(img_gray, axis=2) / 255
                img = img / 255
                # print(img.shape,mask_mean.shape,img_gray.shape)
                img = np.concatenate([img, mask_mean, img_gray], axis=2)
                #img -= img.mean(axis=(0,1,2),keepdims = True)
                #img /= img.std(axis=(0,1,2),keepdims = True)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            yield x_batch, y_batch

from model.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff,weighted_bce_dice_loss_hans





callbacks = [EarlyStopping(monitor='val_loss',
                           patience=2,
                           verbose=1,
                           min_delta=1e-5),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.2,
                               patience=1,
                               verbose=1,
                               epsilon=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath='weights/best_weights_RMSprop.hdf5',
                             save_best_only=True,
                             save_weights_only=True),
             TensorBoard(log_dir='logs')]
model.load_weights('weights/best_weights_RMSprop.hdf5')
model.compile(optimizer=RMSprop(lr=1e-6), loss = bce_dice_loss, metrics=[dice_coeff])
model.fit_generator(generator=train_generator(),
                    steps_per_epoch=500,#np.ceil(float(len(ids_train_split)) / float(batch_size)),
                    epochs=10,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=valid_generator(),
                    validation_steps=200)#np.ceil(float(len(ids_valid_split)) / float(batch_size)))











try:
    model.summary()
    model.load_weights('weights/best_weights_Adam.hdf5')
    print('loading weights.......succeeded')
except:
    print('loading weights.......failed')


callbacks = [EarlyStopping(monitor='val_loss',
                           patience=2,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.2,
                               patience=1,
                               verbose=1,
                               epsilon=1e-5),
             ModelCheckpoint(monitor='val_loss',
                             filepath='weights/best_weights_Adam.hdf5',
                             save_best_only=True,
                             save_weights_only=True),
             TensorBoard(log_dir='logs')]
model.compile(optimizer=Adam(lr=1e-6), loss=bce_dice_loss, metrics=[dice_coeff])
model.fit_generator(generator=train_generator(),
                    steps_per_epoch=500,#np.ceil(float(len(ids_train_split)) / float(batch_size)),
                    epochs=10,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=valid_generator(),
                    validation_steps=200)#np.ceil(float(len(ids_valid_split)) / float(batch_size)))


callbacks = [EarlyStopping(monitor='val_loss',
                           patience=2,
                           verbose=1,
                           min_delta=1e-5),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.2,
                               patience=1,
                               verbose=1,
                               epsilon=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath='weights/best_weights_sgd.hdf5',
                             save_best_only=True,
                             save_weights_only=True),
             TensorBoard(log_dir='logs')]
model.load_weights('weights/best_weights_sgd.hdf5')
model.compile(optimizer=sgd(lr=1e-6), loss=bce_dice_loss, metrics=[dice_coeff])
model.fit_generator(generator=train_generator(),
                    steps_per_epoch=500,#np.ceil(float(len(ids_train_split)) / float(batch_size)),
                    epochs=10,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=valid_generator(),
                    validation_steps=200)#np.ceil(float(len(ids_valid_split)) / float(batch_size)))
