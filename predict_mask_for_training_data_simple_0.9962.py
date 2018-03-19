import cv2
import numpy as np
import pandas as pd
import threading
import queue
import tensorflow as tf
# from tqdm import tqdm

import params

input_size = params.input_size
batch_size = params.batch_size
orig_width = params.orig_width
orig_height = params.orig_height
threshold = params.threshold
model = params.model_factory()

df_test = pd.read_csv('input/train_masks.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))


# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle

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

rles = []

model.load_weights(filepath='weights/best_weights.hdf5')
graph = tf.get_default_graph()

q_size = 10


def data_loader(q, ):
    for start in range(0, len(ids_test), batch_size):
        print('generating masks for training data....{:02d}% finished'.format(start/len(ids_test)))
        x_batch = []
        end = min(start + batch_size, len(ids_test))
        ids_test_batch = ids_test[start:end]
        for id in ids_test_batch.values:
            img = cv2.imread('input/train/{}.jpg'.format(id))
            mask_mean0 = cv2.imread('input/mask_mean_{}.jpg'.format(id[-2:]), cv2.IMREAD_GRAYSCALE)
            mask_mean = cv2.resize(mask_mean0, (input_size, input_size))
            mask_mean = np.expand_dims(mask_mean, axis=2)

            img = cv2.resize(img, (input_size, input_size))
            img = np.concatenate([img, mask_mean, img * mask_mean], axis=2)
            x_batch.append(img)
        x_batch = np.array(x_batch, np.float32) / 255
        q.put(x_batch)


def predictor(q, ):
    for i in range(0, len(ids_test), batch_size):
        x_batch = q.get()
        with graph.as_default():
            preds = model.predict_on_batch(x_batch)
        preds = np.squeeze(preds, axis=3)
        for pred in preds:
            prob = cv2.resize(pred, (orig_width, orig_height))
            mask = prob > threshold

            cv2.imwrite('predicted_train_masks/'+ids_test[i]+'_mask.png',mask*255)
            rle = run_length_encode(mask)
            rles.append(rle)


q = queue.Queue(maxsize=q_size)
t1 = threading.Thread(target=data_loader, name='DataLoader', args=(q,))
t2 = threading.Thread(target=predictor, name='Predictor', args=(q,))
print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), batch_size))
t1.start()
t2.start()
# Wait for both threads to finish
t1.join()
t2.join()

print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submit/submission_train.csv.gz', index=False, compression='gzip')
