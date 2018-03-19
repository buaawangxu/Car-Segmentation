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
#model = params.model_factory()

df_test = pd.read_csv('input/sample_submission.csv')
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



#model.load_weights(filepath='weights/best_weights.hdf5')
graph = tf.get_default_graph()

for m in [4,5,6,3,2,1]:
    rles = []
    count = 0
    for id in ids_test.values:
        count += 1
        print('Encoding predict masks into rle......'+str(count)+'/'+str(len(ids_test))+'     '+str(count/len(ids_test)))
        mask_optimized = cv2.imread('C:/Users/Whisper-xu/Desktop/kaggle ouput before optimization/0.9963_optimize_'+str(m)+'/'+id+ '_mask.png',cv2.IMREAD_GRAYSCALE)
        mask_optimized = mask_optimized/255
        #print(np.max(mask_optimized),np.min(mask_optimized))
        mask_optimized = mask_optimized > threshold
        rle = run_length_encode(mask_optimized)
        rles.append(rle)
    print("Generating submission file...")
    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv('C:/Users/Whisper-xu/Desktop/kaggle ouput before optimization/submission files/submission_test_optimized_'+str(m)+',.csv.gz', index=False, compression='gzip')