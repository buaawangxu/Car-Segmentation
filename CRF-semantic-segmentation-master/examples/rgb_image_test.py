#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:10:58 2017

@author: sy
"""


import pydensecrf.densecrf as dense_crf
from cv2 import imread
import matplotlib.pyplot as plt
from densecrf2 import crf_model, potentials
import numpy as np
# Create unary potential
unary = potentials.UnaryPotentialFromProbabilities(gt_prob=0.8)

bilateral_pairwise = potentials.BilateralPotential(
    sdims=80,
    schan=13,
    compatibility=4,
    kernel=dense_crf.DIAG_KERNEL,
    normalization=dense_crf.NORMALIZE_SYMMETRIC
)

gaussian_pairwise = potentials.GaussianPotential(
    sigma=3, 
    compatibility=2,
    kernel=dense_crf.DIAG_KERNEL,
    normalization=dense_crf.NORMALIZE_SYMMETRIC
)

# =============================================================================
# Create CRF model and add potentials
# =============================================================================
#zero_unsure:  whether zero is a class, if its False, it means zero canb be any of other classes
# =============================================================================
# crf = crf_model.DenseCRF(
#     num_classes = 3,
#     zero_unsure = True,              # The number of output classes
#     unary_potential=unary,
#     pairwise_potentials=[bilateral_pairwise, gaussian_pairwise],
#     use_2d = 'rgb-2d'                #'rgb-1d' or 'rgb-2d' or 'non-rgb'
# )
# =============================================================================
crf = crf_model.DenseCRF(
    num_classes = 2,
    zero_unsure = False,              # The number of output classes
    unary_potential=unary,
    pairwise_potentials=[bilateral_pairwise, gaussian_pairwise],
    use_2d = 'rgb-2d'                #'rgb-1d' or 'rgb-2d' or 'non-rgb'
)


# =============================================================================
# Load image and probabilities
# =============================================================================
image = imread('im.jpg')
probabilities = imread('label.png')
probabilities[:,:,0]  = 0*probabilities[:,:,0] 
probabilities[500:700,:,1] = probabilities[500:700,:,1]//10*9
probabilities[:,:,2] = 255 - probabilities[:,:,1]
#probabilities = probabilities[:,:,0:2]
print(probabilities.shape,np.max(probabilities),np.min(probabilities))
print(image.shape,np.max(image),np.min(image))

# =============================================================================
# Set the CRF model
# =============================================================================
#label_source: whether label is from softmax, or other type of label.
crf.set_image(
    image=image,
    probabilities=probabilities,
    colour_axis=2,                  # The axis corresponding to colour in the image numpy shape
    class_axis=2,                   # The axis corresponding to which class in the probabilities shape
    label_source = 'label'           # where the label come from, 'softmax' or 'label'
)

# =============================================================================
# run the inference
# =============================================================================
# Run 10 inference steps.
crf.perform_step_inference(5)
mask0 = crf.segmentation_map

# Run 80 inference steps.
crf.perform_step_inference(40)  # The CRF model will restart run.
mask80 = crf.segmentation_map

# Plot the results
plt.subplot(221)
plt.title('Segmentation mask after 10 iterations')
plt.imshow(mask0)

plt.subplot(222)
plt.title('Segmentation mask after 80 iterations')
plt.imshow(mask80)

plt.subplot(223)
plt.title('Segmentation mask after 10 iterations')
plt.imshow(image)

plt.subplot(224)
plt.title('Segmentation mask after 10 iterations')
plt.imshow(probabilities[:,:,1])

plt.show()

'''
crf.perform_inference(10)  # The CRF model will restart run.
new_mask80 = crf.segmentation_map
print(crf.kl_divergence)
plt.imshow(new_mask80)
plt.show()
'''











