# Notes on papers for instance segmentation

## A review on 2D instance segmentation based on deep neural networks
Link: https://www.sciencedirect.com/science/article/pii/S0262885622000300
1. Two-stage top-down method advantages: simple method of extending detection to perform segmentation
2. Two-stage top-down method disadvantages: dependent on object detection results; do not conform to human intuition
3. Two-stage bottom-up method advantages: conform to human intuition
4. Two-stage bottom-up method disadvantages: requires good semantic segmentation backbone; post-processing method has poor generalisation ability
5. Multi-stage method advantage: better performance
6. Multi-stage method disadvantage: no scheme to balance accuracy and computational cost
7. Backbones
* ResNet: Can reach depths of 152 layers, which can extract more information. 
* Feature Pyramid Network: integrates high-level semantic information and low-level localisation information
* Deformable Convolutional Network: deformable convolution (operates on irregular regions) and deformable RoI pooling (non-constant position mapping). Can improve ability to localise non-grid objects.
* Swin Transformer: computes self-attention within a local window and constructs a hierarchical feature representation by merging adjacent patches in deep
