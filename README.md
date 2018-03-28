# Carvana Image Masking Challenge

This is a project for the Kaggle competition "Carvana Image Masking Challenge". The basic idea of this competition is to automatically identify the boundaries of the car in an image.

#Stage One: Search for Useful Ideas

## 1. Baseline    
> As there is only one type of object in this task, the basic idea is using U-net structure for the segmentation.     

## 2. Possible optimazations     
> (1) Data argumentation: Rotation, Shift, Adding noise, Scale    
> (2) Multi-Scale features: Use the idea of DenseNet and Feature Pyramid Net to use more information of features.    
> (3) Add densenet structure at the top to get more precise result at the edge.    
> (4) Add other blocks to further improve the result: inception block, dense block, sen block, capsule block, Diation Convolution.    

## 3. Get more ideas    
> (1) Read papers about optimization of U-net.    
> (2) Read the discussion and kernals of this competition on Kaggle.

#Stage Two: Ensenble Useful Ideas

## 4. Summary of useful ideas     
> (1) Use Diation Convolution in the middle layer.    
> (2) Ensemble results of different argumentation pictures    
> (3) Use another U-net to train along the edges of the predictions of U-net using small pathes of the original image. The ultimate implementation of this idea is to combine these two U-net into one model and train an end-to-end model (share the features like fast rcnn).    

## 5. To do list    
> (1) Use Diation Convolution in the middle layer.    

>          -Finished       CV 0.9941      PB 0.9939  input_shape 256×256×3

> (2) Use another U-net to train along the edges of the predictions of U-net using small pathes

>          -without the prediction patches   CV 0.9765  no improvement on original image    
>          -with the prediction patches      CV 0.9865



 
