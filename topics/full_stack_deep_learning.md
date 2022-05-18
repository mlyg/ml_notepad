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

## Lecture 6
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

## Lecture 7
1. Reasons for poor model performance:
* Implementation bugs
* Hyperparameter choices
* Data/model fit
* Dataset construction
2. He initialisation for ReLU and Glorot for tanh
3. Python's ipdb is useful for debugging
4. Overfitting to a single batch is useful for catching errors:
* Error goes up: often a flip sign somewhere in the loss function/gradient
* Error explodes: usually a numerical issue but can also be caused by a high learning rate
* Error oscillates: learning rate or data (shuffled labels or incorrect data augmentation)
* Error plateaus: learning rate too low or regularisation, or loss function or the data pipeline issue
5. Test error = irreducible error + bias + variance + distribution shift + validation overfitting
6. It can be useful to have a test validation set to compare to the test error to estimate distribution shift
7. Ranked list of hyperparameters and impact: 
* High: learning rate, learning rate scheduler, loss function, layer size
* Medium: weight initialisation, layer params, weight of regularisation
* Low: optimiser, optimiser params, batch size, non-linearity
8. Hyperparameter tuning:
* Manual: can be time-consuming and challenging
* Grid search: sample uniformyl across range, inefficient 
* Random search: empircally performs better than grid search, but results less interpretable
* Coarse-to-fine search: most popular method, gradually narrowing onto best hyperparameter range
* Bayesian hyperparameter optimisation: efficient, switching between training with hyperparameter values that maximise the expected improvement (per the model) and use training results to update the initial probabilistic model and its expectations

## Chapter 8
1. Data is the best way to improve machine learning performance
2. Data flywheel: users contribute to data once model released
3. Semi-supervised learning: training data is automatically labelled by exploiting correlations between different input signals
4. Data augmentation in different domains:
* images: crop, flip, rotate
* tabular: delete some cells to simulate missing data
* text: replace words with synonyms and change order
* speech and video: change speed, insert a pause, mix sequences
5. Population based augmentation is a faster method than AutoAugment to find optimal data augmentation settings
6. Data storage system building blocks:
* filesystem: networked filesystem (NFS) is accessible over the network by multiple machines; distributed file system (HDFS) is stored and accessed over multiple machines
* object storage: API over the filesystem that allows users to use a command on files (GET, PUT, DELETE) to a service without worrying where they are actually stored
* database: PostgreSQL is the right choice most of the time; persistent, fast, scalable storage and retrieval of structured data; fundamental unit is a row
* data warehouse: structured aggregation of data for analysis, known as online analytical processing (OLAP)
* data lake:  unstructured aggregation of data from multiple sources
7. Federated learning trains a global model on several local devices without ever acquiring global access to the data - issues: 
* sending model updates can be expensive
* the depth of anonymization is not clear
* system heterogeneity when it comes to training is unacceptably high

## Lecture 9
1. Ethical theories:
* Divine command: behaviour is moral if the divine commands it - not explored by philosophy
* Virtue ethics: behaviour is moral if it upholds a person's virtues e.g. bravery, generosity. However virtues are not persistent across a person's life and somewhat illusory
* Deontology: behaviour is moral if it satisfies a categorical imperative. Might lead to counter-intuitive moral decisions and unacceptable inflexibility
* Utilitarian: behaviour is moral if it brings the most good to the most people. Utility is difficult to measure
* Rawl's theory of justice: equal distribution of resources
2. Long term AI ethical problems:
* autonomous weapons
* lost human labour: bad if no social safety net and no other jobs for the unemployed. Good because we need labour following the demographic inversion
* human extinction
3. Alignment problem: AI systems built need to be aligned with our goals and values
5. AI for hiring:
* Data is biased and can lead to prejudice
* AI can end up amplifying existing biases
6. Fairness
* Even if you hide the protected attribute, the AI could find other patterns in the data that correlate with it
* tradeoffs between individual fairness and group fairness
7. Representation
* there is a lack of attention to diverse representation in the development of technology products
* this can be addressed by including people from all backgrounds
* and deliberately ensuring products reflect inclusive values
* one challenge is deciding whether the model should learn about the world as it is in the data or learn about the world in a more idealistic manner.
8. Ideas to confront fairness
* perform ethical risk sweeping: engage in regular fairness checks on behalf of different stakeholders
* expand the ethical circle: consider different perspectives than your own
* think about the worse-case scenario
* close the loop: put in place a system that can improve

## Lecture 10
1. Problem with black-box predictions:
* production distribution does not always match the offline distribution
* expected performance does not tell the whole story
* performance of your model is not equal to the performance of your machine learning system
* might not be evaluating the metric that is important in the real world
2. Software types of tests
* unit tests: test functionality of single piece of code in isolation
* integration tests: test how two or more units perform together
* end-to-end tests: tests how the entire software system performs when all units are put together
3. Best practices
* automate tests
* make sure tests are reliable, run fast and undergo code review
* enforce that tests must pass before merging onto current branch
* when a new production bug is found, make sure they become a test
* follow the testing pyramid: from most to least - unit test (70%) -> integration test (20%) -> end-to-end test (10%)
4. Controversial best practices
* solitary tests: does not rely on real data from other units, while sociable testing makes implicit assumption that other modules are working
* test coverage: states the percentage of lines of code in your codebase is called by at least one test. However, test coverage does not measure the right things (in particular, test quality)
* test-driven development: create your tests before you write your code and use tests for specification of code function
5. Testing in production
* Canary deployment: roll out new software to a small percentage of your users and separately monitor that group’s behavior
* A/B testing: run a more principled statistical test if you have particular metrics that you care about: one for the old version of the code that is currently running and another for the new version that you are trying to test.
* Real user monitoring: Rather than looking at aggregate metrics (i.e., click-through rate), try to follow the journey that an actual user takes through your application and build a sense of how users experience the changes
* Exploratory testing: Testing in production is not something that you want to automate fully. It should involve a bit of exploration (individual users or granular metrics)
6. Continuous Integration and Continuous Delivery
* automate tests by hooking into the repo
* Github Actions is easy to integrate
7. Common mistakes while testing ML systems
* Think the ML system is just a model and only test the model
* Not test the data
* Not build a granular enough understanding of the performance of the model before deploying it
* Not measure the relationship between model performance metrics and business metrics
* Rely too much on automated testing
* Think offline testing is enough, and therefore, not monitor or test in production
8. ML system
* training system: takes code and data as inputs and produces the trained model as the output
* prediction system: takes in and pre-processes the raw data, loads the trained ML model, loads the model weights, calls model.predict() on the data, post-processes the outputs, and returns the predictions
* serving system: takes in requests from users, scales up and down to meet the traffic demands, and produces predictions back to those users
* production data: both the predictions that the model produces and additional feedback from users, business metrics, or labelers
* labeling system: takes the raw data seen in production, helps you get inputs from labelers, and provides labels for that data
* storage and pre-processing system: stores and pre-processes the labeled data before passing it back to the training system
9. System component tests
* infrastructure tests: unit tests for training system e.g. single batch or single epoch tests
* training tests: integration tests between data system and training system
* functionality tests: unit tests for prediction system e.g. load a pre-trained model and test its predictions on a few key examples
* evaluation tests: integration tests between your training system and your prediction system. Consider all metrics: model metrics, behavioural metrics, robustness metrics, privacy and fairness metrics and simulation metrics
* shadow tests: integration tests between your prediction system and your serving system. Help detect bugs in the production deployment, such as inconsistencies between offline model/data and production model/data
* A/B tests: determine the impact of different model predictions on user and business metrics
* labelling tests: spot check labels or measure agreement between labellers
* expectation tests: unit tests for data to test for quality
10. Explainable and interpretable AI
* Domain predictability: the degree to which it is possible to detect data outside the model’s domain of competence.
* Interpretability: the degree to which a human can consistently predict the model’s result
* Explainability: the degree to which a human can understand the cause of a decision
11. Interpretable family of models
* simple, familiar models like linear regression, logistic regression, generalized linear models, and decision trees
* attention models: however only tell where model looking, not why. 
12. Distillation
* fit a more complex model and interpret its decision using another model from an interpretable family
* the additional model is called a surrogate model
* however if surrogate model performs well, then unclear why not to just apply the surrogate model directly. Similarly, if it does not perform well, unclear if it genuinely represents model behaviour
* local surrogate models (LIME) focus on a single point to generate an explanation for, by performing local pertubations of the data and training a surrogate model to make similar predictions to complex model
13. Contribution of features to prediction
* partial dependence plots and individual conditional expectation plots
* permutation feature importance: selects a feature, randomizes its order in the dataset, and sees how that affects performance. Easy to use but does not work for high dimensional data or where there is feature interdependence
* Shapley additive explanations (SHAP): SHAP scores test how much changes in a single feature impact the output of a classifier when controlling for the values of the other features
* Gradient-based saliency maps: determines how much does a unit change in the value of the input’s pixels affect the prediction of the model. Similar problem to attention
14. When explainability is needed
* Regulators demand 
* Users demand
* Deployment demand: in this case, domain predictability is the real aim
15. At present, true explainability for deep learning models is not possible
* it can be easy to cherry-pick specific examples that can overstate explainability
* methods tend to be unreliable and highly sensitive to the input
* the full explanation is often not available to modern explainability methods
