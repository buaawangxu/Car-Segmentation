#https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge/blob/master/train.py

import cv2
import numpy as np
import pandas as pd
import threading
import queue
import tensorflow as tf
#from tqdm import tqdm

import params
import h5py


input_size = params.input_size
batch_size = params.batch_size
orig_width = params.orig_width
orig_height = params.orig_height
threshold = params.threshold
model = params.model_factory()

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



def data_loader(q, ):
    for start in range(0, len(ids_test), batch_size):
        x_batch = []
        end = min(start + batch_size, len(ids_test))
        ids_test_batch = ids_test[start:end]
        for id in ids_test_batch.values:
            print('\nTesting Generating...' + id + '    '+str(end) + '/' + str(len(ids_test.values)))
            img = cv2.imread('input/test/{}.jpg'.format(id))
            img = cv2.resize(img, (input_size, input_size))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(40, 40))
            #img_gray = clahe.apply(img_gray)
            img_gray = cv2.equalizeHist(img_gray)
            mask_mean = cv2.imread('input/mask_mean_{}.jpg'.format(id[-2:]), cv2.IMREAD_GRAYSCALE)
            mask_mean = cv2.resize(mask_mean, (input_size, input_size))

            mask_mean = np.expand_dims(mask_mean, axis=2) / 255
            img_gray = np.expand_dims(img_gray, axis=2) / 255
            img = img / 255
            # print(img.shape,mask_mean.shape,img_gray.shape)
            img = np.concatenate([img, mask_mean, img_gray], axis=2)
            #img -= img.mean(axis=(0, 1, 2), keepdims=True)
            #img /= img.std(axis=(0, 1, 2), keepdims=True)

            x_batch.append(img)
        x_batch = np.array(x_batch, np.float32)
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
            #cv2.imwrite('D:/Carvana/CarSegmentation 20170911/CarSegmentation/input/test/{}.jpg'.format(ids_test[i].values),prob*255)
            rle = run_length_encode(mask)
            rles.append(rle)

df_test = pd.read_csv('input/sample_submission.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

names = []
rles = []
for id in ids_test:
    names.append('{}.jpg'.format(id))
model.load_weights(filepath = 'weights/best_weights_RMSprop.hdf5')
graph = tf.get_default_graph()
q_size = 2
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
df.to_csv('submit/submission_RMSprop.csv.gz', index=False, compression='gzip')

names = []
rles = []
for id in ids_test:
    names.append('{}.jpg'.format(id))
model.load_weights(filepath = 'weights/best_weights_Adam.hdf5')
graph = tf.get_default_graph()
q_size = 10
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
df.to_csv('submit/submission_Adam.csv.gz', index=False, compression='gzip')

names = []
rles = []
for id in ids_test:
    names.append('{}.jpg'.format(id))
model.load_weights(filepath = 'weights/best_weights_sgd.hdf5')
graph = tf.get_default_graph()
q_size = 10
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
df.to_csv('submit/submission_sgd.csv.gz', index=False, compression='gzip')
