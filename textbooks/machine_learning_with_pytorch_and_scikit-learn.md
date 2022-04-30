# Machine Learning with PyTorch and Scikit-Learn

## Chapter 1
1. Other names for 'feature' are: predictor, variable, input, attribute and covariate
2. Other names for 'target' are: outcome, output, response variable, dependent variable, (class) label and ground truth

## Chapter 2
1. First artificial neuron was the McCulloch-Pitts (MCP) neuron in 1943: simple logic gate where signals arrive at dendrites, are integrated at cell body and output signal generated if exceeds threshold
2. Rosenblatt perceptron learning rule: automatically learn optimal weight coefficients that are then multipled by input features to decide whether to fire or not
3. The perceptron updates the weights with every training example, and only converges if linearly separable
4. Adaptive linear neurons (Adaline) rule (also known as Widrow-Hoff rule): updates weights using linear activation function (identity) rather than unit step function in perceptron

## Chapter 3
1. The logit function is the logarithm of the odds (p / (1 - p))
2. The only difference between logistic regression and Adaline is the activation function (identity vs sigmoid)
3. there are lots of algorithms for convex optimisation beyond SGD, such as 'newton-cg', 'lbfgs', 'liblinear', 'sag' and 'saga'
4. scikit-learn now uses 'lbfgs' rather than 'liblinear' because it can handle the multinomial loss while 'liblinear' is limited to using one vs all for multiclass classification
5. The parameter 'C' in logistic regression is inversely proportional to the regulatisation parameter lambda
6. Vapnik proposed soft-margin classification for SVM using a slack variable to allow convergence with non-linearly separable cases
7. Gini impurity is similar to, but not the same as, entropy
8. scikit-learn has an automatic cost complexity post-pruning procedure

## Chapter 4
1. pandas dataframe has dropna method, which has some useful parameters: axis (0 = row, 1 = column), how ('all' drops rows where all columns are NaN), thresh (drops rows with fewer than threshold values)
2. one-hot encoding introduces multi-collinearity, and this can be reduced by dropping one of the feature columns (because it is redundant)
3. normalisation usually involves rescaling features to [0, 1], while standardisation involves centering feature columns with mean = 0 and standard deviation = 1
4. RobustScaler (scikit-learn) is useful for scaling when using little data
5. Sequential backward selection (SBS) reduces dimensionality of feature space by eliminating feature that causes the least performance drop once removed
