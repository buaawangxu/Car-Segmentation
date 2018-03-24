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

import pydensecrf.densecrf as dense_crf
from cv2 import imread
import matplotlib.pyplot as plt
from densecrf2 import crf_model, potentials
from model.losses import dice_coeff_np
# Create unary potential
unary = potentials.UnaryPotentialFromProbabilities(gt_prob=0.99)

bilateral_pairwise = potentials.BilateralPotential(
    sdims=10,
    schan=20,
    compatibility=8,
    kernel=dense_crf.DIAG_KERNEL,
    normalization=dense_crf.NORMALIZE_SYMMETRIC
)

gaussian_pairwise = potentials.GaussianPotential(
    sigma=10, 
    compatibility=8,
    kernel=dense_crf.DIAG_KERNEL,
    normalization=dense_crf.NORMALIZE_SYMMETRIC
)

crf = crf_model.DenseCRF(
    num_classes = 2,
    zero_unsure = False,              # The number of output classes
    unary_potential=unary,
    pairwise_potentials=[bilateral_pairwise, gaussian_pairwise],
    use_2d = 'rgb-1d'                #'rgb-1d' or 'rgb-2d' or 'non-rgb'
)


input_size = params.input_size
batch_size = params.batch_size
orig_width = params.orig_width
orig_height = params.orig_height


threshold = params.threshold
model = params.model_factory
model.summary()
model.load_weights(filepath='weights/best_weights.hdf5')

'''
df_test = pd.read_csv('input/train_masks.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

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

        image = cv2.resize(255*x_batch[i,:,:,:],(orig_width, orig_height))
        prob = np.expand_dims(prob,axis = 2)
        prob = 255*np.concatenate([0*prob,prob,255-prob],axis =2)
        prob = prob.astype(np.uint8)
        image = image.astype(np.uint8)
        print(image.shape)
        print(prob.shape)
        crf.set_image(
            image=image,
            probabilities=prob,
            colour_axis=2,                  # The axis corresponding to colour in the image numpy shape
            class_axis=2,                   # The axis corresponding to which class in the probabilities shape
            label_source = 'label'           # where the label come from, 'softmax' or 'label'
        )

        crf.perform_step_inference(3)
        mask_crf = crf.segmentation_map
        print(np.min(mask_crf[:,:,0]),np.max(mask_crf[:,:,0]),np.median(mask_crf[:,:,0]))
        print(np.min(mask_crf[:,:,1]),np.max(mask_crf[:,:,1]),np.median(mask_crf[:,:,1]))
        print(np.min(mask_crf[:,:,2]),np.max(mask_crf[:,:,2]),np.median(mask_crf[:,:,2]))

        #plt.imshow(mask_crf[:,:,0])
        #plt.show()
        #plt.imshow(mask_crf[:,:,1])
        #plt.show()        
        #plt.imshow(mask_crf[:,:,2])
        #plt.show()
        
        cv2.imwrite('input/train_masks_predict/'+ids_test_batch.values[i]+'_mask_crf.png',mask_crf)
        mask_gt = cv2.imread('input/train_masks/{}_mask.png'.format(ids_test_batch.values[i]), cv2.IMREAD_GRAYSCALE)
        mask_gt = cv2.resize(mask_gt, (orig_width, orig_height))
        if len(mask_gt.shape)>2:
            mask_gt = mask_gt[:,:,0]
        dice_coef = dice_coeff_np(1.0*(mask_gt/255>0.5),1.0*(mask>0.5))
        if len(mask_crf.shape)>2:
            mask_crf = mask_crf[:,:,1]
        dice_coef_crf = dice_coeff_np(1.0*(mask_gt/255>0.5),1.0*(mask_crf/255>0.5))
        print('the score with U-net is ',dice_coef)
        print('the score with U-net and CRF is ',dice_coef_crf)

'''

# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle

df_test = pd.read_csv('input/sample_submission.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])
rles = []

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))

for start in range(0, len(ids_test), batch_size):
    print('generating masks for training data....{:02d}/{:02d}finished'.format(start,len(ids_test)))
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
        cv2.imwrite('input/test_masks_predict/'+ids_test_batch.values[i]+'_mask.png',mask*255)
        rle = run_length_encode(mask)
        rles.append(rle)


print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submit/submission_test_without_optimization_1.csv.gz', index=False, compression='gzip')