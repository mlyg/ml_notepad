# Machine Learning with PyTorch and Scikit-Learn

## Chapter 1
1. Other names for '**feature**' are: predictor, variable, input, attribute and covariate
2. Other names for '**target**' are: outcome, output, response variable, dependent variable, (class) label and ground truth

## Chapter 2
1. First artificial neuron was the **McCulloch-Pitts (MCP) neuron** in **1943**: simple logic gate where signals arrive at dendrites, are integrated at cell body and output signal generated if exceeds threshold
2. **Rosenblatt perceptron learning rule**: automatically learn optimal weight coefficients that are then multipled by input features to decide whether to fire or not
3. The perceptron updates the weights with every training example, and only converges if linearly separable
4. **Adaptive linear neurons (Adaline)** rule (also known as **Widrow-Hoff rule**): updates weights using linear activation function (identity) rather than unit step function in perceptron

## Chapter 3
1. The **logit function** is the logarithm of the odds (p / (1 - p))
2. The only difference between logistic regression and Adaline is the activation function (identity vs sigmoid)
3. there are lots of algorithms for convex optimisation beyond SGD, such as **'newton-cg'**, '**lbfgs'**, **'liblinear'**, **'sag'** and **'saga'**
4. scikit-learn now uses 'lbfgs' rather than 'liblinear' because it can handle the **multinomial loss** while 'liblinear' is limited to using one vs all for multiclass classification
5. The parameter **'C'** in logistic regression is **inversely proportional** to the **regulatisation parameter lambda**
6. **Vapnik** proposed **soft-margin classification** for SVM using a **slack variable **to allow convergence with non-linearly separable cases
7. Gini impurity is similar to, but not the same as, entropy
8. scikit-learn has an **automatic cost complexity post-pruning procedure**

## Chapter 4
1. pandas dataframe has **dropna method**, which has some useful parameters: **axis** (0 = row, 1 = column), **how** ('all' drops rows where all columns are NaN), **thresh** (drops rows with fewer than threshold values)
2. **one-hot encoding** introduces **multi-collinearity**, and this can be reduced by dropping one of the feature columns (because it is redundant)
3. **normalisation** usually involves rescaling features to [0, 1], while **standardisation** involves centering feature columns with mean = 0 and standard deviation = 1
4. **RobustScaler** (scikit-learn) is useful for scaling when using little data
5. **Sequential backward selection** (SBS) reduces dimensionality of feature space by eliminating feature that causes the least performance drop once removed

## Chapter 5
1. **Feature selection** maintains the original features, while **feature extraction** transforms the data into a new feature space
2. **Principal component analysis** (PCA) performs **unsupervised** dimensionality reduction
3. The features must be **standardised** prior to use in PCA for equal weight to be assigned to each feature
4. **PCA steps:**
* **Standardise** the dataset (d-dimensions)
* **Construct** the **covariance** matrix
* **Decompose** the **covariance** matrix into eigenvectors and eigenvalues
* **Sort eigenvalues** in **descending order** to rank the corresponding eigenvectors
* Select **k largest eigenvectors** where k < d
* Create a **projection matrix W** from the k eigenvectors
* Transform the d-dimensional input using the projection matrix to produce a new k-dimensional feature space
5. numpy.linalg.**eigh** (rather than numpy.linalg.**eig**) is more **numerically stable** when working with symmetrical matrices and always returns **real** eigenvalues
6. Setting n_components = None using the PCA class, we can get the explained variance ratio with the attribute explained_variance_ratio_
7. We can get the **loadings** (how much each original feature contributes to a given principal component) by multiplying the engivector by the **square root of the eigenvalue** (which gives us a correlation between the feature and principal component)
8. **Linear discriminant analysis** finds the feature space that optimises class separability
9. **LDA assumptions:**
* data is **normally** distributed
* classes have identical covariance matrices
* training examples are statistically independent of one another
10. **LDA steps:**
* **Standardise** the dataset (d-dimensions)
* For each class, compute the **d-dimensional mean vector**
* Create the **between-class scatter matrix** Sb, and **within-class scatter matrix** Sw
* Compute the eigenvectors and eigenvalues of the matrix Sw^-1 Sb
* Sort the eigenvalues by **decreasing order** to rank eigenvectors
* Choose the k eigenvectors that correspond to the k largest eigenvalues to make the transformation matrix W with the eigenvectors as columns of the matrix
* Project the data onto the new feature subspace using W
11. The number of linear discriminants is **at most c - 1**, where c is the number of class labels
12. If there is **perfect collinearity**, then the covariance matrix would have rank 1, and would **only have 1** non-zero eigenvalue
13. Non-linear dimensionality reduction techniques are also referred to as **manifold learning**: a manifold is a lower dimensional topological space embedded in a high-dimensional space
14. **t-distributed stochastic neighbour embedding** (t-SNE): embeds data points into a lower dimensional space such that the pairwise distances in the original dimensional space are preserved
15. The t-SNE embedding should be **initialised with PCA**
16. **Uniform manifold approximation and projection** (UMAP) is typically faster than t-SNE, and can be used to project new data unlike t-SNE

## Chapter 6
1. scikit-learn **make_pipeline** function takes an arbitrary number of scikit-learn **transformers** (objects that has fit and transform methods), followed by a scikit-learn **estimator** (object that has fit and predict methods)
2. A good standard value for k in k-fold cross-validation is **10**
3. **Higher values of k** results in **lower pessimistic bias** towards estimating generalisation performance, but will **increase runtime** and yield **higher variance** estimates
4. **Stratified k-fold cross-validation** can yield better bias and variance estimates especially when there are unequal class proportions, by preserving class label proportions in each fold
5. **High bias** is where the model has both **low training and cross-validation accuracy** -> can be addressed by increasing model parameters or decreasing degree of regularisation
6. **High variance** is where there is a **large gap between training and cross-validation accuracy** -> can be addressed by reducing model parameters or increasing degree of regularisation
7. **plt.fill_between** can be used to show error
8. **Grid search** performs a brute-force exhaustic search 
9. **Randomised search** can find more optimal hyperparameters than grid search
10. **Halving random/grid search** can be a more efficient method for searching hyperparameters
11. **Tree-structured Parzen Estimators (TPE)** is a Bayesian optimisation method based on probabilistic model that uses past hyperparameter evaluations to find better hyperparameter configurations
12. **Nested cross-validation** is useful when evaluating different algorithms: outer k-fold cross-validation loop splits data into training and test folds, and the inner loop is used to select the model using k-fold cross-validadtion on the training fold
13. **Matthews correlation coefficient (MCC)** ranges between -1 and 1, and takes all elements of a confusion matrix into account (including TN which F1 does not account for)
14. **For multiclass problems**
* **Micro-average** is calculated from individual TPs, TNs, FPs and FNs: useful to weight each prediction equally
* **Macro-average** is calculated as the average scores of different systems: weights all classes equally
15. **Weighted macro-average** which weights score of each class label by number of true instances is used as a default in scikit-learn
16. **Synthetic Minority Over-sampling Technique (SMOTE)**: a way of dealing with class imbalance by generating synthetic training examples

## Chapter 7
1. **Plurality voting:** the generalisation of majority voting principle to multiclass settings
2. Ensembling improves performance if the classifiers have equal error rates and are independent, and as long as the base classifiers perform better than random guessing
3. **Bagging**: bootstrap samples are used to train individual classifiers
4. Random forests are a special case of bagging where random feature subets are used when fitting individual decision trees
5. **Bagging** can **reduce variance** but **not reduce bias**, which is why unprunced decision trees which have low bias are often used
6. **Boosting** involves ensembling weak learners, focusing on training examples that are hard to classify
7. **Original boosting algorithm:**
* Draw a random sample of training examples, d1, **without replacement** from the training dataset, to train a weak learner C1
* Draw a second random training subset, d2, without replacement from the training dataset and add 50% of the misclassified examples to train a weak learner C2
* Find training examples d3 in the training set which C1 and C2 disagree on to train a third weak learner C3
* Combine weak learners C1, C2 and C3 by majority voting
8. **Boosting** can lead to a **reduction in both bias and variance**, but are known for their high variance
9. **AdaBoost** is slightly different from the original boosting algorithm by using the complete dataset to train weak learners, and instead weighting training examples to focus on misclassified examples
10. **Gradient boosting**: iteratively fits decision trees using prediction errors
11. Gradient boosting trees are typically deeper than AdaBoost, and use the prediction error directly to form a target variable for fitting the next tere (rather than assigning sample weights)
12. Gradient boosting involves calculating **pseudo-residuals**, and fitting subsequent trees to minimise the pseudo-residuals

## Chapter 8
1. **Bag-of-words** model allows us to represent text as numerical feature vectors:
* First a **vocabulary** of all the words from the set of documents is created
* Each **document** is represented as a **feature vector** which contains the **word frequency** for each word in the vocabulary
2. A bag is created using **CountVectorizer** from sklearn.feature_extraction.text 
3. The values in the feature vectors are called raw term frequencies: tf(t, d) - the number of times a term t occurs in a document d
4. The **order** of terms in a bag-of-words model **does not matter**
5. Bag-of-words model is also called a 1-gram (unigram) model because each token in the vocabulary represents a single word - n-grams of size 3 or 4 appear to be better for spam-filtering
6. **Term frequency-inverse document frequency** (tf-idf): downweights frequently occuring words, by multiplying tf(t,d) by idf(t,d), where idf(t,d) = log (nd/1+df(d,t))
7. **Porter stemmer algorithm**: tokenises words by their stem
8. **Stemming** can produce non-real words, while **lemmatisation** aims to obtain more grammatically correct form of indivdual words - however little difference in performance has been observed
9. **Stop words** include 'is', 'and', 'has' which are frequent and do not contain useful information
10. The 'liblinear' solver can perform better than 'lbfgs' for large datasets
11. **Out-of-core learning**: memory efficient method involving mini-batching to train models
12. **word2vec**: a more moden alternative to bag-of-words, which uses an unsupervised learning algorithm based on neural networks to learn the relationship between words, and can reproduce words using simple vector math
13. **Topic modeling**: the task of assigning topics to unlabeled text documents
14. **Latent Dirichlet allocation**: generative probabilistic model that tries to find groups of words that appear frequently together across documents
* Input is a bag-of-words matrix, and **decomposes** this into a **document-to-topic** matrix and **word-to-topic** matrix
15. The **number of topics** must be specified as a **hyperparameter** for LDA

## Chapter 9
1. Linear regression does **not** require the target variable to be normally distributed
2. A **correlation matrix** is a standardised version of the covariance matrix, and contains the **Pearson product-moment correlation coefficient** (Pearson's r) which measures the linear dependence between pairs of features
3. The **bias** variable is **0** if the features are **standardised**
4. Scikit-learn linear regression works better with **unstandardised** variables
5. **Random Sample Consensus (RANSAC) algorithm**: fits a robust regression model
* Select a random number of examples to be inliers and fit the model
* Test all other data points against fitted model and add those points that fall within a specific tolerance
* Refit the model using all inliers
* Estimate the error of the fitted model against the inliers
* Terminate the algorithm if either the performance reaches a certain threshold or a fixed number of iterations reached
6. By default, scikit-learn uses the **median absolute deviation (MAD)** estimate to select the inlier threshold
7. By plotting **residuals**, we would expect them to be randomly scattered if there was additional information to be captured
8. **Mean squared error** and **mean absolute error** are unbounded, and interpretation depends on dataset and feature scaling
9. **Coefficient of determination (R^2)**: the standardised version of MSE, and is the fraction of response variance captured by the model, and is equal to: 1 - SSE/SST (SSE: sum of squared errors, SST: total sum of squares i.e. variance of target variable)
10. **Ridge regression**: L2 regularisation where the sum of squared weights are added to MSE. The bias b is not regularised. 
11. **LASSO**: L1 regularisation where sum of absolute weights are added to MSE. Can be used for feature selection, but it selects at most n features if m > n, where n is the number of training examples
12. **Elastic net**: Both L1 and L2 regularisation
13. **Polynomial regression**: includes polynomial terms to model non-linear relationships. Importantly, it still uses linear regression coefficients.
14. There is a trade-off between complexity of the model (overfitting) and bias
15. A **random forest** can be thought of as a **sum of piecewise linear functions** (in cotrast to global linear/polynomial regression)
16. Decision trees analyse one feature at a time, and do not take weighted combinations, which explains why standardisation is not necessary
17. For decision tree regression, the MSE is referred to as **within-node variance**, and therefore he splitting criterion is better known as **variance reduction**

## Chapter 10
1. **k-means clustering belongs to the category of prototype-based clustering**: each cluster is represented by a protoscopy, which is either the **centroid** (average) for continuous features or **medoid** (point that minimise distance to all other points of a cluster) for categorical features
2. **k-means** is good for identifying **spherical** cluster shapes
3. k-means requires that the **number of clusters** are **specified**, and one or more clusters can be empty
4. **k-means algorithm:**
* randomly pick k number of centroids from examples as initial cluster centres
* assign each example to the nearest centroid
* move centroid to centre of clusters
* assign each example again to nearest centroid and repeat until either cluster assignments do not change, user-defined tolerance reached or maximum number of iterations reached
5. The **Euclidean distance metric** is commonly used as the similarity metric, and can be solved by **minimising the within-cluster SSEs**, which is sometimes called cluster inertia
6. Features should be scaled if the Euclidean distance metric is used
7. **k-means++ algorithm**:
* initialise an empty set M to store the k centroids being selected
* randomly choose the first centroid and assign it to M
* For each example not in M, find the minimum squared distance to any of the centroids in M
* To randomly select the next centroid, use a weighted probability distribution favouring points further from centroids
* Repeat until steps k centroids chosen
8. k-means clustering is an example of **hard clustering**
9. **Fuzzy C-means algorithm** is an example of soft clustering: it is similar to k-means but each point is assigned with a probability of belonging to each cluster
10. There is an additional **fuzziness coefficient** (fuzzifier, m), which controls the degree of fuzziness. The larger the value of m, the smaller the cluster membership becomes
11. The **within-cluster SSE (distortion)** is used to quantify the quality of clustering. The **elbow** method uses distortion to estimate the optimal numbero f clusters k.
12. The **silhouette coefficient** is also useful to quantify the quality of clustering:
* Calculate the **cluster cohesion**: average distance between an example and all other points in the same cluster
* Calculate the **cluster separation**: average distance between the example and all examples in the nearest cluster
* Calculate the **silhouette**: cluster separation - cluster cohesion / max(cluster separation, cluster cohesion)
13. Values range between -1 and 1
14. **Hierarchical clustering**: two main approaches are agglomerative and divisive hierarchical clustering
15. Advantages of hierarchical clustering: can plot **dendrograms** to help with interpretation, and **do not need to specify** the number of clusters upfront
16. The two standard algorithms for **agglomerative hierarchical clustering** are **single** linkage and **complete** linkage: single linkage computes distances between the most similar members for each pair of clusters and merges the two clusters which have the smallest distance. In contrast, the complete linkage approach compares the two most dissimilar members to form the merge
17. Other types of agglomerative hierarchical clustering include **average** linkage (merge cluster pairs based on minimum average distance between all group members) and **Ward** linkage (merge the two clusters that lead to the minimum increase of total within-cluster SSE)
* Compute pair-wise distance matrix of all examples
* Represent each data point as its own cluster
* Merge the two closest clusters based on distance between most distant members
* Update cluster linkage matrix
* Repeat until one single cluster remains
18. **Density-based spatial clustering of applications with noise (DBSCAN)**: does not make spherical assumption like k-means, and does not partition dataset into hierarchies that require a manual cut-off:
* A **core** point is a point that has at least a specified number of neighbouring points fall within the specified radius
* A **border** point is a point that has fewer neighbours than the specified number, but lies within the specified radius of a core point
* All other points are considered **noise points**
* After labelling all points, form separate clusters for each core point or connected group of core points
* Assign each border point to the cluster of its corresponding core point
19. Disadvantages of DBSCAN: has two hyperparameters, does not work well with high dimensional data

## Chapter 11
1. It common to scale pixel values to [-1,1] 
2. **Backpropagation** is a computationally efficient approach to computing** partial derivatives of complex, non-convex loss functions**. The partial derivatives are used to learn weight coefficients for parameterising multilayer artificial neural networks
3. **Automatic differentiation** has two modes: forward and reverse (backpropagation is a special case of the reverse)
4. Backpropagation is efficient because it involves **matrix-vector multiplication** rather than matrix-matrix multiplication

## Chapter 12
1. By default, Python is limited to execution on one core due to the **global interpreter lock (GIL)**
2. **CUDA** and **OpenCL** are special packages to use the GPU
3. The Pytorch **computation graph** is defined **implicitly** rather than being explicitly constructed and executed
4. **torch.tensor()** creates a tensor from a list, and torch.from_numpy creates a tensor from a numpy array
5. **torch.to()** used to change tensor type
6. **torch.chunk()** divides an input tensor into a list of equally sized tensors: arguments - chunk: number of splits, dim=dimension which to split over
7. **torch.split()** is similar by divides tensor into specified sizes
8. **torch.stack()** needs same dimension to stack, while torch.cat() does not
9. **torch.utils.data.DataLoader()** provides an automatic and customisable batching to a dataset
10. **Dataset** class must contain the __init__(), __len__() and __getitem__() methods
11. **nn.Module** allows layers to be stacked to form a network
12. The **sigmoid function** does not perform well if the input is highly negative, because the output will be close to zero, and neural network training will be slow, and more likely to be trapped in a local minima. For this reason, a **hyperbolic tangent** is often preferred as an activation function in hidden layers
13. The hyperbolic tangent (tanh) can be seen as a rescaled version of the logistic function. The benefit is that it has a broader output spectrum ranging from [-1,1] which can improve the convergence of the backpropagation algorithm
14. The logistic function is available in **scipy.special expit**
15. **ReLU** helps with the vanishing gradient problem because its derivative with respect to its input is always 1 for positive input values (but becomes very small for sigmoid and tanh)

## Chapter 13
1. PyTorch performs its computations based on a **directed acyclic graph (DAG)**
2. Only tensors of **floating point** and **complex** dtype can require gradients
3. requires_grad is set to False by default, and can be set to true efficiently by using the method **requires_grad_()** [in place method]
4. **Initialising random weights** is necessary to break the symmetry during backpropagation - otherwise multilayer neural networks would be the same as a single-layer neural network
5. **Xavier** initialisation balances the **variance of the gradients across different layers**, and takes into account the number of input neurons and the number of output neurons in a layer. This performed better than just usin ga random uniform or random normal weight initialisation
6. **Automatic differentiation**: implementation of the chain rule for computing gradients of nested functions
7. The **gradient** of a function is a vector composed of all the inputs' partial derivatives
8. **Forward** accumulation traverses the chain rule from inside out, while **reverse** accumulation traverses the chain rule from outside in
9. nn.Sequential is useful to build models in a cascaded way
10. **Universal approximation theorem**: a feedforward neural network with a single hidden layer and a relatively large number of hidden units can approximate arbitrary continuous functions relatively well
11. **nn.Embedding** maps each index to a vector of random numbers of type float which can be trained
12. The training_step, training_epoch_end, validation_step, test_step and configure_optimizers methods are specific to **PyTorch lightning**
13. **PyTorch Ignite** has a training loop abstraction (unlike PyTorch which uses two for loops), additional training metrics and built-in handlers to compose training pipelines, save artifacts, and log parameters and metrics

## Chapter 14
1. Each feature map element comes from a **local** patch of pixels in the input
2. CNNs perform well on image-related tasks because of **sparse connectivity** and **parameter sharing**, making the assumption that pixels nearby are more relevant to each other
3. A **discrete convolution** is denoted by y = x * w, where x is the **input/signal** and w is the **filter/kernel**
4. The computation involves first **rotating** the filter (not the transpose), and then computing the **dot product** in a sliding window approach
5. **Cross-correlatio**n is similar to convolution but the filter does not need to be rotated. PyTorch implements cross-correlation, but refers to it as convolution
6. Padding is important to prevent middle elements from contributing more. **Types of padding:**
* **full**: p = m - 1. This increases the dimension of the output and is rarely used, except to minimise boundary effects in signal processing applications
* **same**: ensures output vector has the same size as the input vector, and is the **most common**
* **valid**: p = 0
7. Convolution operations are efficiently implemented, for example using **Fourier transforms**
8. Max-pooling introduces **local invariance**, helping to generate features robust to noise
9. Pooling also decreases the size of features, which results in higher computational efficiency
10. Setting stride of 2 in convolutional layers results in a pooling layer with learnable weights
11. PyTorch format: **Tensor[batches, channels, image_height, image_width]**
12. The convolution operation is performed on all channels separately and then summated
13. L2 regularisation and weight_decay are equivalent when using stochastic gradient descent
14. **Dropout** forces the network to learn a redundant representation of the data, which enables it to learn more general and robust patterns from the data
15. Providing logits rather than class membership probabilities for computing the cross entropy loss is preferred due to numerical stability reasons
17. RMSProp and AdaGrad inspired Adam

## Chapter 15
1. Typical supervised learning algorithms assume the input is **independent and identically distributed **
2. The **order matters** for modelling sequence data
3. Time series data is a special type of sequential data where each sample is taken at successive time points
4. The different categories of **sequence modelling**:
* **Many-to-one**: input data is a sequence, but output is a fixed size vector e.g. sentiment analysis
* **One-to-many**: input data is in standard format but the output is a sequence e.g. image captioning
* **Many-to-many**: both input and output are sequences, and either synchronised (video classification) or delayed (translation)
5. In an RNN, the hidden layer receives its input from both the input layer of the current time step and hidden layer from the previous time step
6. The flow of information is usually displayed as a loop, also known as a **recurrent edge** in graph notation
7. A single-layer RNN has one hidden layer
8. RNNs are trained using** backpropagation through time (BPTT)**
9. Besides hidden recurrence, there is also **output to hidden** recurrence and **output to output** recurrence
10. BPTT involves a multiplicative factor in computing gradients of a loss function leading to vanishing/exploding gradient problem. There are three solutions to this:
* Gradient clipping
* Truncated backpropagation through time (TBPTT)
* LSTM
11. **Long short-term memory (LSTM) cells:**
* contain a **recurrent edge** with w = 1 to overcome vanishing/exploding gradient problems. The values associated with this edge is called the **cell state**
* the cell state is modified for the next time step without being multiplied directly by any weight factor
* the flow of information through the memory cell is controlled by **gates**
* **forget gate**: allows memory cell to reset the cell state
* **input gate** (sigmoid) and **candidate value** (tanh) are responsible for updating the cell state
* **output gate**: decides how to update the values of the hidden unit
12. **Feature embedding** maps each word to a vector of fixed size with real-valued elements, and has two main advantages over one-hot encoding
* reduction in dimensionality of feature space
* extraction of salient features
13. **Bidirectional LSTMs** make recurrent layer pass through input sequences from both directions, making a forward and reverse pass over the input sequence, with the resulting hidden state of the forward and reverse pass usually concatenated into a single hidden state
14. **Autoregression**: using the generated sequence as input for generating new elements

## Chapter 16
1. A limitation of seq2seq with RNN is that the **single hidden unit must remember the entire input sequence**, and compressing all the information into a single hidden unit may cause **loss of information**
2. An **attention mechanism** allows the RNN **access to all input elements** at each time step, but assigns **different attention weights** to each input element
3. **Bidirectional RNNs** are useful because current inputs may have a dependence on sequence elements that came either before or after it in a sentence, or both
4. Attention-based RNN consists of two RNNs: the first RNN generates a **context vector** (weighted version of the concatenated hidden states) from input sequence elements, and the second RNN receives the context vector as input
5. The attention weight is a normalised version of the **alignment score**, which evaluates how well the input matches the output
6. Self-attention models the dependencies of the input element to all other input elements:
* Firstly, **importance weights** are derived based on the similarity between the current element and all other elements in the sequence
* Secondly, the weights are **normalised**
* Thirdly, the weights are used in combination with corresponding sequence element to compute the attention value
7. **Scaled dot-product attention** has learnable parameters that allow attention values to change during model optimisation
8. There are three weight matrices, whose name comes from **information retrieval systems and databases**, where a query is matched against key values for which certain values are retrieved:
* **Query** sequence q
* **Key** sequence k
* **Value** sequence v
9. Alongside applying softmax, the attention coefficients are normalised by **1/sqrt(m)** to ensure that the **Euclidean length** of the weight vectors will be approximately in the same range
10. **Attention is all you need transformer:**
* The encoder takes the sequential input and maps it into a **continuous representation** that is then passed to the decoder
* The encoder is a stack of 6 identical layers, which each contain two sublayers: **multi-head self-attention** and **fully-connected layer**
* Multi-head attention uses **multiple heads** (sets of query, value and key matrices) similar to how CNNs use multiple kernels
* In practice, rather than having a separate matrix for each attention head, transformer implementations use a single matrix for all heads, accessing logically separate regions of the matrix using masks
* The vectors are concatenated into one long vector and a linear projection (fully connected layer) is used to map it back to an ppropriate length
* The decoder further contains a **masked** multi-head attention sublayer, to mask out words later in the sentence
* The main difference between the encoder and decoder is that the encoder calcluates attention across all words in a sentence (bidirectional input parsing), while the decoder only considers elements that are preceding the current input position (unidirectional input parsing)
11. Scaled-dot product attention and fully connected layers are **permutation invariant**, and therefore positional encodings are important, which involve adding a vector of small values to the input embeddings at the beginning of the encoder and decoder blocks e.g. **sinusoidal encoding** (which prevents the positional encoding from getting too large)
12. **Layer normalisation**: preferred in NLP, where sentence lengths can vary. It computes normalisation statistics across all feature values independently for each training example, relaxing minibatch size dependencies
13. A simple way to perform **self-supervised learning** (unsupervised pretraining) is to perform **next-word prediction** on a large corpus, enabling the model to learn the probability distribution of words and can form a strong basis for becoming a powerful language model
14. Using a pretrained model, transfer learning is done either by:
* **Feature-based approach**: uses pre-trained representations as additional features to a labelled dataset, and train a classifer using the embedding outputs as features
* **Fine-tuning approach**: add a fully connected layer to the pre-trained model to accomplish tasks
16. **Generative Pre-trained Transformer (GPT): **
* GPT-1 uses a transformer decoder structure, relying on preceding words to predict the next word
* GPT-2 does not require any additional modification during input or fine-tuning stages unlike GPT-1
* GPT-3 shifts focus from zero to **few shot learning** via in-context learning, and uses sparse attention by only attending to a subset of elements with limited size (rather than the O(n^2) with dense attention
17. **Bidirectional Encoder Representations from Transformers (BERT):**
* Uses a transformer encoder
* Uses bidirectional training procedure, which prevents it from generating a sentence word by word, but provides input encodings of higher quality for other tasks such as classification
* BERT not only uses positional embeddings and token embeddings, but also **segment embeddings** which is used for a special pre-training task called next-sentence prediction (given two sentences, must decide whether it belongs to the first or second sentence)
* BERT pretraining is either masked language modelling (predict hidden words) or next-sentence prediction (predict whether a sentence is the next sentence or not)
19. **Bidirectional and Auto-regressive Transformer (BART):**
* GPT specialty is generating text, whereas BERT performs better on classification
* BART can be seen as a generalisation of both GPT and BERT, and can both generate and classify text, using a bidirectional encoder as well as left-to-right autoregressive decoder
* Pretraining BART involves feeding **corrupted input** (token masking, deletion, text infilling, sentence permutation and document rotation) and it must learn to reconstruct the input
* BART performs summarisation rather than sequence generation

## Chapter 17
1. First proposed in **2014** by **Ian Goodfellow**
2. Purpose is to **synthesise new data** that has the **same distribution** as its training dataset
3. **Autoencoders** do not synthesise new data, but perform **data compression** by mapping the input data onto a **latent vector**
4. An autoencoder without any non-linearity is like PCA, except that **PCA** has an additional **orthonormal constraint**
5. Deep autoencoders have multiple layers and non-linearities
6. Autoencoders with a latent vector dimension smaller than input is called **undercomplete**
7. **Overcomplete** audoencoders have a latent vector larger than input, and can be used for noise reduction, by adding random noise to input (**denoising autoencoder**)
8. A **variational autoencoder** is a generalisation of autoencoders into a generative model
* With an input sample x, the encoder computes **two moments** of the distribution of the latent vector (mean and variance), and during training the network is forced to match these to the standard normal distribution
* After VAE trained, **encoder is discarded** and the decoder can be used to generate new examples by feeding random z vectors from the ‘learned’ Gaussian distribution
9. Traditionally, generative models could perform **conditional inference**, which involves sampling a feature xi conditioned on another feature xj
10. A GAN model trains the **generator** and **discriminator** models together, and improve by playing an adversarial game
11. The loss function is the value function, which can be interpreted as a payoff: **maximise value** with respect to **discriminator** and **minimise value** with respect to **generator**. GANs are trained by alternating between the two optimisation steps by freezing parameters of each model in turn
12. **ReLU** results in **sparse gradients** which may not be suitable when we want to have gradients for the full range of input values, therefore **leaky ReLU** is often used
13. **Deep convolutional GAN (DCGAN)** first uses a fully connected layer to project the random vector to be reshaped into a spatial convolution representation, and then uses as series of transposed convolutions to upsample the image to the desired output image size
14. **Transposed convolution** is also known as **fractionally strides convolution**, and is not a deconvolution because it focuses on recovering the dimensionality of the feature space and not the actual values. It works by adding zeros between the elements and then applying a convolution
15. Batch normalisation involves normalising layer inputs and preventing changes in their distribution during training, which enables faster and better convergence. **Batch normalisation involves: **
* Compute **mean** and **standard deviation** of the net inputs for each mini-batch
* **Standardise** the net inputs for all examples in the batch
* **Scale** and **shift** the normalised net inputs using two learnable parameter vectors gamma and beta
* During training, the running averages and variance are computed
16. Batch normalisation was initially developed to help reduce **internal covariance shift** (change that occurs in distribution of a layer’s activation due to updating network parameters), but recent research suggests that batch normalisation mainly helps by **smoothening** the surface of the loss function
17. It is **not usually recommended** to use bias units in layers that follow batch normalisation because bias units would be redundant
18. **Dissimilarity measures between two distributions**
* **Total variation**: measures the largest difference between the two distributions at each point. sup(S) refers to the smallest value that is greater than all elements of S i.e. the least upper bound. inf(S) used in Earth mover’s distance refers to the largest value that is smaller than all elements of S i.e. greatest lower bound. 
* **Earth mover’s distance**: minimal amount of work needed to transform one distribution into another
* **Kullback-Leibler (KL)** and **Jensen-Shannon (JS) divergence**: come from information theory. KL is not symmetric i.e. KL(P||Q) != KL(Q||P)
19. EM distance has advantages over KL, TV and JS, and can be simplified using the **Kantorovich-Rubinstein duality theorem**
20. **WGAN** uses the discriminator as a critic rather than a classifier, output scalar scores rather than probability values. To ensure the **1-Lipschitz property** is preserved, the weights are clamped to a small region e.g. [-0.01,0.01]
21. However, it was found that clipping weights led to exploding/vanishing gradients, and capacity underuse, therefore WGAN with **gradient penalty** (WGAN-GP) was proposed
22. WGAN uses layer normalisation rather than batch normalisation
23. **Mode collapse** is a common issue with GAN, where the generator gets stuck in a small subspace and generates similar samples - **mini-batch discrimination**, **feature matching** and **experience replay** can help overcome this

## Chapter 18
1. **Undirected graph** G is a pair (V,E) where V is the set of graph nodes and E is the set of edges making up the paired nodes. The graph can be encoded as a |V| x |V| **adjacency matrix** A where 1 denotes an edge between the nodes I and j
2. **Directed graphs** are defined like undirected graphs but E consists of the set of **ordered pairs** of edges
3. **Labeled graphs** use feature matrices to contain formation about node or edges
4. Graphs have **structural locality** compared to locality in 2D space in images
5. A strict prior for graph data is **permutation invariance**: ordering of nodes does not affect the output
6. A convolutional approach is desirable for graphs because it can function with a fixed parameter set for graphs of different sizes
7. **Message-passing framework** of graph convolutions: each node has an associated hidden state, and each graph convolution is split into a **message-passing** and **node update** phase
8. **Spectral graph convolutions** are different to spatial graph convolutions, and are useful for capturing **global patterns** in graph data. Spectral graphs operate using the graph’s spectrum (its set of eigenvalues) by computing the eigendecomposition of a normalised version of the graph’s adjacency matrix called the graph Laplacian
9. For an undirected graph, the **Laplacian matrix is L = D - A**, where D is the degree matrix (diagonal whose elements on the diagonal is the number of edges in and out of the node on that row)
10. L is real-valued and symmetric, and can therefore be decomposed into L = Q L QT
11. **Spectral convolutions have several undesirable properties: **
* Eigendecomposition is O(n^3)
* Can only be applied to graphs of the same size
* Receptive field is fixed to the whole graph
12. **Chebyshev graph convolutions** can approximate the original spectral convolution at a lower time complexity and can have receptive fields with varying sizes
13. **There are different types of pooling for graphs:**
* Mean or max pooling
* DiffPool: learns a soft cluster assignment, and performs clustering and downsampling simultaneously
* Top-k pooling: drops nodes from graphs
14. There are different types of **normalisation**
* GraphNorm
* MessageNorm

## Chapter 19
1. RL is not about teaching how to do things, only what we want the agent to achieve
2. **Dynamic programming vs recursion**: DP stores the results of subproblems so that they can be accessed in constant time
3. If the dynamics of the environment are known, then DP can be used to solve it, otherwise TD or MC its needed
4. A **Markov process** can be represented as a **directed cyclic graph** where the nodes represent the state and edges the transition probabilities - the transition probabilities coming out of each state sum to 1
5. With DP, we can perform one-step look ahead to find the action that gives the maximum value
6. For the replay buffer, using a **deque data structure** from Python collections is more efficient than a list
