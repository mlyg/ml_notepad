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
