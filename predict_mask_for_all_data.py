import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import params
from keras.losses import binary_crossentropy
import keras.backend as K
import numpy as np
from scipy.spatial.distance import pdist, cdist
import cv2


input_size = params.input_size
batch_size = params.batch_size
orig_width = params.orig_width
orig_height = params.orig_height
threshold = params.threshold
model = params.model_factory
model.summary()

df_test = pd.read_csv('input/train_masks.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

model.load_weights(filepath='weights/best_weights.hdf5')

for start in range(0, len(ids_test), batch_size):
    print('generating masks for training data....{:02d} % finished'.format(int(100*start/len(ids_test))))
    x_batch = []
    end = min(start + batch_size, len(ids_test))
    ids_test_batch = ids_test[start:end]
    for id in ids_test_batch.values:
        img = cv2.imread('input/train/{}.jpg'.format(id))
        img = cv2.resize(img, (input_size, input_size))
        x_batch.append(img)
    x_batch = np.array(x_batch, np.float32) / 255


    preds = model.predict_on_batch(x_batch)
    preds = np.squeeze(preds, axis=3)
    for i,pred in enumerate(preds):
        prob = cv2.resize(pred, (orig_width, orig_height))
        mask = prob > threshold
        cv2.imwrite('input/train_masks_predict/'+ids_test_batch.values[i]+'_mask.png',mask*255)


df_test = pd.read_csv('input/sample_submission.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])


for start in range(0, len(ids_test), batch_size):
    print('generating masks for training data....{:02d} % finished'.format(int(100*start/len(ids_test))))
    x_batch = []
    end = min(start + batch_size, len(ids_test))
    ids_test_batch = ids_test[start:end]
    for id in ids_test_batch.values:
        img = cv2.imread('input/test/{}.jpg'.format(id))
        img = cv2.resize(img, (input_size, input_size))
        x_batch.append(img)
    x_batch = np.array(x_batch, np.float32) / 255

    preds = model.predict(x_batch)
    preds = np.squeeze(preds, axis=3)
    for i,pred in enumerate(preds):
        prob = cv2.resize(pred, (orig_width, orig_height))
        mask = prob > threshold
        cv2.imwrite('input/train_masks_predict/'+ids_test_batch.values[i]+'_mask.png',mask*255)