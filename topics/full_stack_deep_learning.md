# Full stack deep learning spring 2021

## Lecture 1
1. **Universal function approximation theorem**: given any continuous function f(x), if a 2 layer neural network has enough hidden units, then there is a choice of weights that allows it to closely approximate f(x)
2. The huber loss is less sensitive to outliers than the mean squared error. It is a combination of the mean squared error and absolute value error
3. Well conditioned data has zero mean and equal variance in all dimensions
4. Conditioning methods:
* First order methods: weight initialisation, normalisation
* Second order methods: Newton's method, Natural gradient, Adagrad, Adam, Momentum
5. The goal of back propagation is to calculate partial derivatives with respect to any weight or bias
6. For backpropagation to be applied:
* The cost function should be able to be written as an average over the training examples
* The cost function can be written as a function of the output of the neural network
7. The Hadamard/Schur product involves element-wise multiplication

## Lab 1
1. watch -n1 nvidia-smi is useful to watch GPU on google colab
2. by default, ndarrays are float64
3. plt.matshow has the origin at the top left corner, plt.imshow has the origin at the bottom left corner
4. The module pdb defines an interactive source code debugger for Python programs
5. importlib library is useful for dynamic importing using importlib.import_module()
6. '->' is known as a function annotation, and it means the type of the result the function returns
7. @classmethod: used to create class methods that are passed to the actual class object much like self is used for the class instance. It can be used as another way to instantiate an object.
8. @staticmethod: does not contain 'self', so does not have access to instance data. Often a helper/utility method. Truly static in nature.

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

## Lecture 3
1. Feedforward networks cannot deal with variable length scaling, and has memory requirement scaling linearly in number of timesteps. It is also overkill as it has to learn patterns everywhere that may occur in the sequence
2. The core idea of RNN is stateful computation
3. Problem with rNN: all the information in the input sequence is condensed into one hidden state vector
4. Vanilla RNNs cannot handle more than 10-20 timesteps because of vanishing gradients
5. ReLU RNNs often have exploding gradients, while sigmoid/tanh leads to vanishing gradient as derivatives << 1
6. LSTMs work well for most tasks, but worth trying GRUs if LSTMs are not performing well
7. Stacking LSTM layers help with underfitting but are hard to train
8. Attention; instead of compressing all past time steps into a single hidden state, given the neural network access to the entire history, but pay attention only to a subset of past vectors
9. The Google Neural Machine Translation approach used Stacked LSTM encoder-decoder architecture with residual connections, attention and bidirectional, trained using standard cross-entropy loss
10. CTC loss: Model can produce either characters or epsilon token. First merge repeat characters. Then remove any epsilon tokens. The remaining characters are the output
11. RNN training is not as parallelisable as FC or CNN due to sequential nature
12. WaveNet is a convolutional sequential model applied to sequential data. Uses 1D causal convolution because the entire window is sampled from the past. Specifically dilated causal convolutions to increase receptive field. Although training is parallel, inference is serial and slow

## Lecture 4
1. PyTorch can freeze layers by setting .eval()
2. Self attention without learnable parameters involves the dot product and softmax:
* for each input vector, multiply it by the transpose of all other vectors, and then perform softmax to get weights
* multiply weights by all other vectors and sum to get output vector
4. Self attention is permutation invariant
5. For learnable weights:
* The input vector x_i is compared to every other vector to compute attention weights for its output y_i (query)
* The input vector x_i is compared to every other vector to compute attention weights w_ij for output y_j (key)
* Summed with other vectors to form the result of the attention weighted sum (value)
6. T5: text-to-text transfer transformer. Encoder-decoder architecture, evaluated the most recent transfer learning, and input and output are both text strings

## Lecture 5
1. 85% of AI projects fail
2. Robot systems are split into state estimation and control for data efficiency and interpretability
3. Lifecycle of ML project
* Planning and project setup: decide on project, requirements, goals, resource allocation and ethics
* Data collection and labelling: collect training data and annotate
* Model training and debugging: implement baseline, SOTA, debug
* Model deployment and testing: pilot the model in constrained environment, then roll for production
4. Prioritising projects
* High impact
* Cheap prediction
* Product needs
* ML strength
* Inspiration from others
* High feasibility: the three main cost drivers are data availability, accuracy requirement and problem difficulty
5. ML archetypes
* Software 2.0: augmenting rule-based/deterministic software with probabilistic ML
* Human in the loop: output of the model is reviewed by a human before execution into the real world
* Autonomous system: the system engages in decisions which are almost never reviewed by humans
6. With multiple metrics, commonly n-1 metrics are thresholded and one is optimised

## Chapter 6
1. The ML code portion in a real-world ML system is a lot smaller than the infrastructure needed for its support
2. The ML infrastructure:
* Data: data sources, lakes/warehouses, data processing, data exploration, data versioning, data labelling
* Training/evaluation: compute sources, resource management, software engineering, frameworks and distributed training libraries, experiment management, hyperparameter tuning
* Deployment: continuous integration and testing, edge deployment, web deployment, monitoring and feature store
3. Problems with notebooks: 
* challenging for good versioning: notebooks are large json files
* notebook IDE has no integration, lifting or code-style correction
* hard to structure code
* out-of-order execution artefacts
* difficult to run long or distributed tasks
4. Streamlit is an easy way to create an app from a machine learning workflow
5. NVIDIA GPUS Kepler -> Pascal -> Volta -> Turing -> Ampere
6. Volta/Turing preferred due to mixed-precision over Kepler/Pascal
7. Allocating resources to users:
* SLURM: standard cluster job scheduler
* Docker/Kubernetes: Docker packages the dependency stack into a container, and Kubernetes runs containers on cluster
* Custom ML software: AWS sagemaker, paperspace gradient, Determined AI
8. JAX focuses primarily on fast numerical computation with autodiff and GPUs across machine learning use cases
9. Distributed training: process to conduct a single model training process
* Data parallelism: split batch evenly across GPUs - simple
* Model parallelism: split weights across GPUs and pass data through each to train the weights - complex
10. Experiment management
* Tensorboard
* MLFlow
* Weights and biases/neptune
11. Hyperparameter tuning
* SigOpt
* Weights and biases
12. Amazon Sagemaker and Google Cloud Platform and all-in-one solutions
