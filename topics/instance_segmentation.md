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

## K-Net: Towards Unified Image Segmentation
Link: https://arxiv.org/abs/2106.14855
1. Unifies **semantic**, **instance** and **pan-optic** segmentation
2. Uses a set of **dynamic kernels** to assign each pixel to either a potential instance or semantic class
3. Uses **bipartite matching strategy** to assign learning targets for each kernel, building a one-to-one mapping between kernels and instances of an image
4. Box-free and NMS-free

## Cellpose: a generalist algorithm for cellular segmentation
Link: https://www.nature.com/articles/s41592-020-01018-x
1. Generate topological maps using **simulated diffusion**
2. Neural network then trained to predict **horizontal** and **vertical** gradients, as well as binary map 
3. Horizontal and vertical gradients form **vector fields**, where **gradient tracking** is used to assign cells
4. **Global average pooling** on smallest convolutional map to get 'style' of image

## Omnipose: a high-precision morphology-independent solution for bacterial cell segmentation
Link: https://www.biorxiv.org/content/10.1101/2021.11.03.467199v3
1. Uses **distance field** to define a new flow field within Cellpose framework
2. Distance field defined by the **eikonal equation** which has benefits:
* Gradient has **unit magnitude** making it more numerically stable
* Distance field is independent of morphology and topology
* Flow field points uniformly from cell boundaries towards cell centre, coinciding with the **medial axis**, allowing pixels to remain spatially clustered after Euler integration, which addresses the issue of oversegmentation in Cellpose

## Transformers in Medical Imaging: A Survey
Link: https://arxiv.org/abs/2201.09873
1. Convolutions operate locally and provide translational equivariance
2. Transformers efficiently encode long-range dependencies

## Boundary-aware Transformers for Skin Lesion Segmentation
Link: https://arxiv.org/abs/2110.03864
1. Boundary points set is produced using a conventional edge detection algorithm
2. For each point, a circle is drawn with radius (set to 10 as default) and the proportion p of the lesion area in the circle is calculated
3. Larger or smaller proportions indicate that the boundary is not smooth
4. NMS is used to filter points with larger proportion than neighbour k points
5. Filtered points are mapped to patch labels and locations are set to 1 and others set to 0

## GT U-Net: A U-Net Like Group Transformer Network for Tooth Root Segmentation
Link: https://arxiv.org/abs/2109.14813
1. Fourier Descriptor is a quantitative representation of closed shapes independent of their starting point, scaling, location, and rotation
2. Fourier Descriptor (FD) loss function is shape-sensitive and makes use of shape prior knowledge

## A Multi-Branch Hybrid Transformer Network for Corneal Endothelial Cell Segmentation
Link: https://arxiv.org/abs/2106.07557
1. Gets edge map ground truth using canny operator
2. Generates complementary 'body' map by inverting the ground truth and applying a Gaussian blur

## COTR: Convolution in Transformer Network for End to End Polyp Detection
Link: https://arxiv.org/abs/2105.10925
1. Uses bipartite matching loss to search an optimal bipartite matching between predictions with a fixed size of N and ground truth objects 

## Dynamic convolution: Attention over convolution kernels
Link: https://arxiv.org/abs/1912.03458
1. Traditional approaches of convolutions use static convolutional kernels but dynamic network architecture (layers, channels etc)
2. **Dynamic convolutions** instead use dynamic convolutional kernels and static network architecture
3. Instead of one convolutional kernel per layer, dynamic convolutions use **K parallel convolutional kernels** that are aggregated
4. The aggregation of parallel convolution kernels makes output channels shared, with no effect on network width or depth
5. The **squeeze-and-excite** method is used to compute attention
6. Optimising dynamic convolutional kernels is difficult

## How Do Vision Transformers Work?
Link: https://openreview.net/forum?id=D78Go4hVcxO
1. Constraining the MSA to local windows can help learn strong representations
2. MSAs improve the predictive performance of CNNs and ViTs predict well-calibrated uncertainty
3. MSAs flatten the loss landscape leading to improved performance and generalisation, but they allow negative Hessian eigenvalues when trained with little data which leads to non-convexity
4. MSAs aggregate feature maps while Convs diversify them
5. MSAs are low pass filters while Convs are high-pass filters
