# Notes on papers for instance segmentation

## A review on 2D instance segmentation based on deep neural networks
Link: https://www.sciencedirect.com/science/article/pii/S0262885622000300
1. **Two-stage top-down method advantages**: simple method of extending detection to perform segmentation
2. **Two-stage top-down method disadvantages**: dependent on object detection results; do not conform to human intuition
3. **Two-stage bottom-up method advantages**: conform to human intuition
4. **Two-stage bottom-up method disadvantages**: requires good semantic segmentation backbone; post-processing method has poor generalisation ability
5. **Multi-stage method advantage**: better performance
6. **Multi-stage method disadvantage**: no scheme to balance accuracy and computational cost
7. **Backbones**
* **ResNet**: Can reach depths of 152 layers, which can extract more information. 
* **Feature Pyramid Network**: integrates high-level semantic information and low-level localisation information
* **Deformable Convolutional Network**: deformable convolution (operates on irregular regions) and deformable RoI pooling (non-constant position mapping). Can improve ability to localise non-grid objects.
* **Swin Transformer**: computes self-attention within a local window and constructs a hierarchical feature representation by merging adjacent patches in deep

## Nucleus segmentation: towards automated solutions
Link: https://www.sciencedirect.com/science/article/pii/S0962892421002518
1. The U-Net is incorporated in **Cellpose** and **StarDist**, and extended in nnU-Net and U-Net++
2. The U-Net can perform instance segmentation with post-processing
3. **nucleAIzer** uses **Mask R-CNN**
4. **MultiStar** and **SplineDist** extend U-Net based StarDist to enable segmentation of overlapping objects
5. **NuSeT** combines regional proposal network, U-Net and watershed post-processing to segment crowded cells

## Mask R-CNN
Link: https://arxiv.org/abs/1703.06870
1. Extends Faster R-CNN by adding a mask branch for predicting segmentation masks on each RoI, in parallel with the existing branch for classification and bounding box regression
2. The mask branch predicts a segmentation mask for each RoI
3. **RoIAlign** layer preserves spatial location
4. Faster R-CNN consists of two stages: first stage is the **Region Proposal Network** which proposes candidate object bounding boxes, and the second stage uses RoIPool from each candidate box and performs classification and bounding box regression
5. Backbone architecture nomenclature: **network-depth-features** (e.g. ResNet-50-C4) 

## Cell Detection with Star-convex Polygons
Link: https://arxiv.org/abs/1806.03535
1. NMS can be problematic if the objects of interest are poorly represented by their **axis-aligned bounding boxes**, which can be the case for cell nuclei
2. Rather than using axis-aligned bounding boxes, StarDist predicts a **star-convex polygon** for every pixel
3. For each pixel, the distance is regressed to the boundary of the object to which it belongs, along a set of **n predefined radial directions** with equidistant angles
4. The model predicts for every pixel whether it belongs to an object, and use** NMS to select polygon proposals** from pixels with sufficiently high object probability
5. Object probabilities are defined as the **normalised Euclidean distance** to nearest background pixel
6. Star-convex polygon distances are computed as the Euclidean distance to the object boundary by following a radial direction k
7. The polygon distance output layer has the number of channels equal to the number of radial directions, and is optimised with the **mean absolute error** 
8. Uses greedy NMS to retain polygons in a certain region

## SOLO: Segmenting Objects by Locations
Link: https://arxiv.org/abs/1912.04488
1. Learn to segment objects by **locations using instance categories**
2. Using the **centre coordinate** of an object, an object instance can be assigned to one of the grid cells as its centre location category
3. Each output channel is responsible for one of the centre location categories, and the corresponding channel map should predict the instance mask of the object belonging to that location
4. To distinguish instances of different sizes, an FPN is used to assign objects of different sizes to different levels
5. CNNs can implicitly learn the absolute position information from zero-padded operation, but this implicitly learned position information is coarse and inaccurate
6. **CoordConvs** involves an additional two channels that contain the x and y coordinates to give spatial variance
7. **Decoupled SOLO** is an efficient and equivalent variant in accuracy of SOLO

## SOLOv2: Dynamic and Fast Instance Segmentation
Link: https://arxiv.org/abs/2003.10152
1. **SOLO is limited by three bottlenecks:**
* inefficient mask representation and learning
* not high enough resolution for finer mask prediction
* slow mask NMS
2. Uses dynamic instance segmentation: rather than generating one instance mask with SxS channels, generates **Mask Kernel G** and **Mask Feature F** which are decoupled and separately predicted
3. Uses **Matrix NMS** rather than traditional NMS which is 9x faster
