# Full stack deep learning spring 2021

## Lecture 1
1. **Universal function approximation theorem**: given any continuous function f(x), if a 2 layer neural network has enough hidden units, then there is a choice of weights that allows it to closely approximate f(x)
2. The huber loss is less sensitive to outliers than the mean squared error. It is a combination of the mean squared error and absolute value error
3. Well conditioned data has zero mean and equal variance in all dimensions
4. Conditioning methods:
* First order methods: weight initialisation, normalisation
* Second order methods: Newton's method, Natural gradient, Adagrad, Adam, Momentum

## Lab 1
1. watch -n1 -nvidia-smi is useful to watch GPU on google colab
2. by default, ndarrays are float64
3. plt.matshow has the origin at the top left corner, plt.imshow has the origin at the bottom left corner
4. The module pdb defines an interactive source code debugger for Python programs

## Lecture 2
1. The issue with FC layers for images:
* poor scaling with image size
* requires a large number of parameters
* not translation invariant
2. Stacking convolutions increase receptive field
3. Max pooling has fallen out of favour
4. A 1x1 convolution corresponds to applying an MLP to every pixel in the convolutional output
5. AlexNet usually drawn as two parts as the GPU only had 3GB which could not fit the single network
6. AlexNet innovated ReLU, dropout and heavy data augmentation
7. VGGNet only used 3x3 convolutions, and max pooling
8. GoogLeNet (InceptionNet): just as deep as VGG by only 3% of parameters. No fully connected layers. Additional classifier outputs in the middle of the network (deep supervision)
9. "Inception hypothesis": cross channel correlations and spatial correlations decoupled and can be mapped separately (1x1 convolution can only see depth and not spatial correlation)
10. ResNet: deeper networks suffer with the vanishing gradient problem. Used stride for downsampling rather than max pooling
11. DenseNet: more skip connections than ResNet
12. ResNeXt: combining inception and ResNet
13. SENet: adding module of global pooling and FC layer to adaptively reweight feature maps
14. SqueezeNet: uses 1x1 filters to prevent number of channels from increasing
15. Localisation: output the single object's class and its location
16. Detection: output every object's class and location. 
17. Localisation does not scale with multiple objects as do not know in advance how many objects - can slide a classifier
18. YOLO/SSD: put a fixed grid over an object, and predict the object centrepoint and several object anchor points. Predict class and bounding box for each anchor box and cell, then use NMS
19. Region proposal methods only look at interesting regions of image (rather than sliding window)
20. Mesh-RCNN has a voxel branch which outputs a 3D mesh
21. Pose estimation detects joint location 
22. White box attacks have access to model parameters, black box do not
23. Style transfer: constraint on style and constraint on content
