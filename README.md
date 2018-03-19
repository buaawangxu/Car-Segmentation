# Car-Segmentation
kaggle car segmentation

This is a project for the Kaggle competition "Cavena ****". The basic idea of this competition is segmenting the foreground and backgroud of a image which contains a car.

1. Baseline
   As there is only one type of object in this task, the basic idea is using U-net structure for the segmentation. 


2. Possible optimazations
   (1) Data argumentation: Rotation, Shift, Adding noise, Scale
   (2) Multi-Scale features: Use the idea of DenseNet and Feature Pyramid Net to use more information of features.
   (3) Add densenet structure at the top to get more precise result at the edge.
   (4) Add other blocks to further improve the result: inception block, dense block, sen block, capsule block.

3. Get more ideas
   (1) Read papers about optimization of U-net.
   (2) Read the discussion and kernals of this competition on Kaggle.
 
