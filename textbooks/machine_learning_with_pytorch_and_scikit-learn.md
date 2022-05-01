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
* **Standardise** the dataset (d0dimensions)
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
