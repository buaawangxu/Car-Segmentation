import cv2
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from model.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff,weighted_bce_dice_loss_hans
from keras.optimizers import RMSprop,sgd
import matplotlib.pyplot as plt
import params

input_size = params.input_size
epochs = params.max_epochs
batch_size = params.batch_size

df_train = pd.read_csv('input/train_masks.csv')
ids_train = df_train['img'].map(lambda s: s.split('.')[0])



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


def randomShiftScaleRotate(image, mask,mask_mean,mask_std,
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

        mask_std = cv2.warpPerspective(mask_std, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                        borderValue=(
                                            0, 0,
                                            0,))
    return image, mask, mask_mean,mask_std


def randomHorizontalFlip(image, mask,mask_mean,mask_std,u=0.1):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        mask_mean = cv2.flip(mask_mean, 1)
        mask_std = cv2.flip(mask_std,1)
    return image, mask, mask_mean,mask_std
def random_mosaic(img,mask_mean,mask):
    if np.random.random() < 0.1:
        img = img*(1-mask)# + 0.5*mask_mean*np.random.rand(mask_mean.shape[0],mask_mean.shape[1],mask_mean.shape[2])+0.5*mask*np.random.rand(mask_mean.shape[0],mask_mean.shape[1],mask_mean.shape[2])
        mask = mask*0
    return img,mask

def train_generator(ids_train_split):
    while True:
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]
            for id in ids_train_batch.values:
                print('\nTraining Generating...'+id +'/'+str(len(ids_train_batch.values)))
                mask_mean0 = cv2.imread('input/mask_mean_{}.jpg'.format(id[-2:]), cv2.IMREAD_GRAYSCALE)
                mask_mean0 = cv2.resize(mask_mean0, (input_size, input_size))

                mask_std= cv2.imread('input/mask_std_{}.jpg'.format(id[-2:]), cv2.IMREAD_GRAYSCALE)
                mask_std = cv2.resize(mask_std, (input_size, input_size))

                img = cv2.imread('input/train/{}.jpg'.format(id))
                img = cv2.resize(img, (input_size, input_size))
                mask = cv2.imread('input/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (input_size, input_size))

                img = randomHueSaturationValue(img,
                                               hue_shift_limit=(-50, 50),
                                               sat_shift_limit=(-5, 5),
                                               val_shift_limit=(-15, 15))
                img, mask,mask_mean ,mask_std= randomShiftScaleRotate(img, mask,mask_mean0,mask_std,
                                                   shift_limit=(-0.0625, 0.0625),
                                                   scale_limit=(-0.1, 0.1),
                                                   rotate_limit=(-5, 5))
                #img, mask,mask_mean,mask_std = randomHorizontalFlip(img, mask, mask_mean,mask_std)
                img = img /255
                mask = np.expand_dims(mask, axis=2)/255
                mask_mean = np.expand_dims(mask_mean, axis=2)/255
                mask_std = np.expand_dims(mask_std, axis=2)
                img = np.concatenate([img,mask_mean,img*mask_mean,img*(1-mask_mean)],axis = 2)
                #img,mask = random_mosaic(img, mask_mean, mask)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            yield x_batch, y_batch

def valid_generator(ids_valid_split):
    while True:
        for start in range(0, len(ids_valid_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_valid_split))
            ids_valid_batch = ids_valid_split[start:end]
            for id in ids_valid_batch.values:
                print('\nTraining Generating...' + id + '/' + str(len(ids_valid_split.values)))
                mask_mean = cv2.imread('input/mask_mean_{}.jpg'.format(id[-2:]), cv2.IMREAD_GRAYSCALE)
                mask_mean = cv2.resize(mask_mean, (input_size, input_size))

                mask_std = cv2.imread('input/mask_std_{}.jpg'.format(id[-2:]), cv2.IMREAD_GRAYSCALE)
                mask_std = cv2.resize(mask_std, (input_size, input_size))

                img = cv2.imread('input/train/{}.jpg'.format(id))
                img = cv2.resize(img, (input_size, input_size))
                mask = cv2.imread('input/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)

                mask = cv2.resize(mask, (input_size, input_size))

                #img = randomHueSaturationValue(img,
                #                               hue_shift_limit=(-50, 50),
                #                               sat_shift_limit=(-5, 5),
                #                              val_shift_limit=(-15, 15))
                #img, mask, mask_mean, mask_std = randomShiftScaleRotate(img, mask, mask_mean0, mask_std,
                #                                                        shift_limit=(-0.0625, 0.0625),
                #                                                        scale_limit=(-0.1, 0.1),
                #                                                        rotate_limit=(-0, 0))
                #img, mask, mask_mean, mask_std = randomHorizontalFlip(img, mask, mask_mean, mask_std)
                img = img / 255
                mask = np.expand_dims(mask, axis=2) / 255
                mask_mean = np.expand_dims(mask_mean, axis=2) / 255
                mask_std = np.expand_dims(mask_std, axis=2)
                img = np.concatenate([img, mask_mean, img * mask_mean,img * (1 - mask_mean)], axis=2)
                x_batch.append(img)
                y_batch.append(mask)

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            yield x_batch, y_batch

model = params.model_factory((1024,1024,10))
#model.load_weights('weights/best_weights.hdf5')
for id in range(1,17):
    ids_train_id = ids_train[id-1:-1:16]
    ids_train_split, ids_valid_split = train_test_split(ids_train_id, test_size=0.2, random_state=42)
    print('Training on {} samples'.format(len(ids_train_split)))
    print('Validating on {} samples'.format(len(ids_valid_split)))
    model.compile(optimizer=RMSprop(lr=1e-4), loss = weighted_bce_dice_loss, metrics=[dice_coeff])
    callbacks = [EarlyStopping(monitor='val_loss',
                               patience=8,
                               verbose=1,
                               min_delta=1e-4),
                 ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=4,
                                   verbose=1,
                                   epsilon=1e-4),
                 ModelCheckpoint(monitor='val_loss',
                                 filepath='weights/best_weights_{:02d}.hdf5'.format(id),
                                 save_best_only=True,
                                 save_weights_only=True),
                 TensorBoard(log_dir='logs')]
    g = train_generator(ids_train_split)
    x_temp,y_temp = next(g)
    #for k in range(x_temp.shape[3]):
    #    plt.imshow(x_temp[0,:, :,k])
    #   plt.show()
    #plt.imshow(y_temp[0,:,:,0])
    #plt.show()
    model.fit_generator(generator=g,
                        steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=valid_generator(ids_valid_split),
                        validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))
