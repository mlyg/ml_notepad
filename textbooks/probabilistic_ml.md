
## Chapter 1
1. Probabilistic ML means treating all unknown quantities as random variables
2. A probabilistic approach is the optimal approach for decision making under uncertainty
3. Design matrix: the N x D matrix containing input features. Big data means N >> D. Wide data means D >> N
4. Featurisation: the process of converting a variable sized input into a fixed-size feature representation
5. Empirical risk minimisation: finding the parameters of a model that minimises the empirical risk (average loss of the predictors on the training set)
6. The empirical risk is equal to the misclassification loss when a zero-one loss is used
7. Epistemic/model uncertainty: lack of knowledge of input-output mapping
8. Aleotoric/data uncertainty: intrinsic/irreducible stochasticity in the mapping
9. An affine function is a linear function with a bias term
10. The maximum likelihood estimate are the parameters which minimise the negative log likelihood
11. Generalisation gap is the difference between the population risk and empirical risk. We approximate the population risk using the test risk
12. In a probabilistic framework, supervised learning fits a conditional model which specifies a distribution over the inputs, while unsupervised learning fits an unconditional model which can generate new data
13. Factor analysis is similar to linear regression, but only observe the outputs (input features) and not the inputs (latent factors)
14. Self-supervised learning can be used to learn useful features of the data while avoiding infering true latent factors
15. The unconditional negative log likelihood can be used to evaluate unsupervised learning algorithms - treats unsupervised learning as density estimation, and a good model should not be surprised by 'actual' data samples
16. The better performance of CNNs than humans represents better fine-grained classification, rather than better at vision than humans
17. Feature crosses can capture interaction effects which one-hot encoding cannot
18. Missing data occurs in three ways:
* Missing completely at random (MCAR): missingness does not depend on hidden or observable features i.e. missingness not related to features
* Missing at random (MAR): missingness does not depend on hidden features, but may depend on observable features i.e. missingness can be explained by features we can observe
* Not missing at random (NMAR: missingmess may depend on either hidden or observable features - need to model missing data as may be informative
19. Alignment problem: potential discrepancy between what we ask the algorithm to optimise and what we actually want them to do. One proposed solution is to use inverse reinforcement learning where the agent infers the reward by observing human behaviour 

## Chapter 2
1. Epistemic uncertainty: also known as model uncertainty
2. Aleotoric uncertainty: also known as data uncertainty, irreducible
3. Random variables can be constant such as the indicator function
4. The probability density function is the derivative of the cumulative density function
5. The inverse of the cdf is called the inverse cdf/percent point function (ppt)/quantile function
9. Unconditional independence means p(X,Y) = p(X)p(Y), while conditional independence means p(X,Y|Z) = p(X|Z)p(Y|Z)
10. Linearity of expectation: E[aX+b] = aE[x]+b
11. V[aX+b] = a^2V[X]
12. Law of iterated expectations/law of total expectation: E[X]=EY[E[X|Y]] i.e. take weighted average of subpopulations
13. Law of total variance/conditional variance formula: V[X]=EY[V[X|Y]]+VY[E[X|Y]] i.e. the overall variance of a random variable X can be evaluated as the sum of the within-sample and between-sample of X sampled on another random variable Y.
14. Bayes' rule is prior distribution (p(H=h)) multiplied by the likelihood (p(Y=y|H=h)) normalised by the marginal likelihood (p(Y=y))
15. The likelihood is a function since y is fixed, while it is called the observation distribution if y not fixed
16. Inverse probability: inferring the state of the world from observations of outcomes. Bayes' theorem solves the inverse probability problem (posterior distribution)
17. The binomial distribution is a generalisation of the Bernoulli distribution for multiple events. The binomial coefficient is the formula for calculating combinations (where order does not matter, in contrast to permutations)
18. The logit function is the inverse of the sigmoid function, with domain [0,1]
