
## Chapter 1
1. Probabilistic ML means treating all unknown quantities as **random variables,** with a probability distrubition that specifies the **weighted set of possible values**
2. A probabilistic approach is the optimal approach for **decision making under uncertainty**
3. **Design matrix**: the N x D matrix containing input features. Big data means N >> D. Wide data means D >> N
4. **Featurisation**: the process of converting a variable sized input into a fixed-size feature representation
5. **Empirical risk minimisation**: finding the parameters of a model that minimises the **empirical risk** (average loss of the predictors on the training set)
6. The empirical risk is equal to the misclassification loss when a **zero-one loss** is used
7. **Epistemic/model uncertainty**: lack of knowledge of input-output mapping
8. **Aleotoric/data uncertainty**: intrinsic/irreducible stochasticity in the mapping
9. An affine function is a linear function with a bias term
10. L(theta|O) = P(O|theta). The likelihood function is conditioned on the observed data and is a function of the unknown parameters theta.
11. The **maximum likelihood estimate** are the parameters which minimise the negative log likelihood
12. The **NLL is proportional to the MSE**, meaning computing the MLE minimises the MSE
13. A **linear model** induces an MSE loss function that has a **unique global optimum**
15. **Generalisation gap** is the difference between the **population risk** and **empirical risk.** We approximate the population risk using the test risk
16. A **latent variable** cannot be observed, but can be detected by their effect on the variation of observed data. Specific values cannot be obtained, but their **distribution** can be **estimated**
18. In a probabilistic framework, supervised learning fits a conditional model which specifies a distribution over the inputs, while unsupervised learning fits an unconditional model which can generate new data
19. **Factor analysis** is similar to linear regression, but only observe the outputs (input features) and not the inputs (latent factors)
20. Self-supervised learning can be used to learn useful features of the data while avoiding infering true latent factors
21. The **unconditional negative log likelihood** can be used to evaluate unsupervised learning algorithms - treats unsupervised learning as density estimation, and a good model should not be surprised by 'actual' data samples
22. A **probability density function** (pdf) calculates the probability for a specific outcome of a random variable
23. **Density estimation** involves inferring the probability distribution based on a sample of data 
24. **Kernel density estimation (KDE)** is the most common method of nonparametric approach for estimating a pdf for a continuous random variable. Has two parameters: **smoothing parameter/bandwidth** which controls the window of samples used to estimate the probability for a new point, and **basis function** (kernel) which controls the contribution of samples in the dataset toward estimating the probability of a new point
25. The better performance of CNNs than humans represents **better fine-grained classification**, rather than better at vision than humans
26. **Feature crosses** can capture interaction effects which one-hot encoding cannot
27. **Term frequency inverse document frequency (TF-IDF)**: reduces impact of words that occur many times across all documents. Multiplies how frequent a word appears in a document (term frequency) by the inverse of how frequent the word appears across all documents (inverse document frequency)
28. **Byte-pair encoding**: form of data compression that creates new symbols to represent common substrings, avoiding the need to use the <unk> token for unknown words. Common words are represented as single tokens, while rare words are broken down into two or more substrings
29. Missing data occurs in three ways:
* **Missing completely at random (MCAR)**: missingness does not depend on hidden or observable features i.e. missingness not related to features
* **Missing at random (MAR)**: missingness does not depend on hidden features, but may depend on observable features i.e. missingness can be explained by features we can observe
* **Not missing at random (NMAR)**: missingmess may depend on either hidden or observable features - need to model missing data as may be informative
30. **Alignment problem**: potential discrepancy between what we ask the algorithm to optimise and what we actually want them to do. One proposed solution is to use inverse reinforcement learning where the agent infers the reward by observing human behaviour 

## Chapter 2
1. **Epistemic uncertainty**: also known as model uncertainty
2. **Aleotoric uncertainty**: also known as data uncertainty, irreducible
3. The main hypothesis in **active learning** is that if a learning algorithm can choose the data it wants to learn from, it can perform better than traditional methods with substantially less data for training.
4. Random variables can be constant such as the indicator function
5. The **probability density function** is the **derivative** of the cumulative density function
6. The inverse of the cdf is called the **inverse cdf/percent point function** (ppt)/**quantile** function. The cdf a function that takes a value of the random variable x and outputs the fraction of x values below that value. The inverse cdf takes a fraction of values and outputs the x value for which the x values below that value correspond to the fraction of data. 
7. Unconditional independence means p(X,Y) = p(X)p(Y), while conditional independence means p(X,Y|Z) = p(X|Z)p(Y|Z)
8. Independence variables: P(A|B) = P(A). 
9. **Linearity of expectation**: E[aX+b] = aE[x]+b
12. V[aX+b] = a^2V[X]
13. **Law of iterated expectations/law of total expectation**: E[X]=EY[E[X|Y]] i.e. take weighted average of subpopulations
14. **Law of total variance/conditional variance formula**: V[X]=EY[V[X|Y]]+VY[E[X|Y]] i.e. the overall variance of a random variable X can be evaluated as the sum of the within-sample and between-sample of X sampled on another random variable Y.
15. **Bayes' rule** is prior distribution (p(H=h)) multiplied by the likelihood (p(Y=y|H=h)) normalised by the marginal likelihood (p(Y=y))
16. The **likelihood** is a **function** since y is fixed, while it is called the observation distribution if y not fixed
17. **Inverse probability**: inferring the state of the world from observations of outcomes. Bayes' theorem solves the inverse probability problem (posterior distribution)
18. The **binomial distribution** is a generalisation of the Bernoulli distribution for multiple events. The binomial coefficient is the formula for calculating combinations (where order does not matter, in contrast to permutations)
19. The logit function is the inverse of the sigmoid function, with domain [0,1]
20. The **multinomial distribution** generalises the binomial distribution to more than two outcomes, and generalises the categorical distribution to more than one event
21. The softmax function is the multinomial logit
22. The term '**temperature**' related to the softmax function comes from the Boltzmann distribution which has the same form as the softmax function
23. **Log-sum-exp trick**: numerical stability trick to avoid exponentiating large numbers in the softmax formula, and applied to the cross entropy loss by modifying it to accept logits rather than probabilities. LSE works because we can shift the values in the exponent by an arbitrary constant c while still computing the same final value, and if we set c = max{x1...xn} then the largest positively exponentiated term is exp(0) = 1
24. The **precision** of a Gaussian is the **inverse of the variance**
25. The **probit** function is the **inverse of the cdf of a Gaussian**
26. **Homoscedastic regression**: variance is **fixed** and independent of input. Opposite of heteroscedastic regression
27. The **softplus** ensures that the resultant output is **non-negative**
28. The **Gaussian distribution** is the most popular distribution in statistics because:
* two easy to interpret parameters
* **Central limit theorem**: good for modelling residual errors i.e. noise
* makes the least number of assumptions (i.e. maximum entropy) subject to the constraint of a specified mean and variance
29. The Dirac delta function is what happens if the variance of the Gaussian approaches 0, making an infinitely tall spike
30. The **sifting property** of the Dirac delta function is important: Dirac delta function shifted by an amount -x. Multiplying the Dirac delta function by f(y) and integrating, gives the value of f(y) at the x
31. The **Student t-distribution** is an alternative to the Gaussian distribution that is robust to outliers. This is because the probability density **decays as a polynomial function of the squared distance from the centre**, compared to the exponential function in a Gaussian distribution, so there is more probability mass (heavy tails) at the ends. **v is the number of degrees of freedom**, where higher v makes it approach a Gaussian distribution
32. The **Cauchy/Lorentz distribution** is the Student distribution with v = 1. This has **very heavy tails** compared to the Gaussian
33. **Heavy-tailed distributions** are probability distributions whose tails are not exponentially bounded
34.The half Cauchy distribution is folded over on itself so all its probability density are positive real values
34. The Laplace distribution (also known as the double sided exponential distribution) also has heavy tails
35. Both the Student t and Laplace distribution are used for robust linear regression
36. The Beta distribution is constrained to [0,1]. a = b = 1 gives a uniform distribution, a > 1, b > 1 gives unimodal distribution
37. The Gamma distribution is constrained to x > 0, and defined in terms of the shape (a) and rate (b) parameters. Sometimes scale (1/b) is used.
38. Special cases of the Gamma distribution:
* Exponential distribution: shape = 1, describes the times between events in a Poisson process
* Chi-squared distribution: shape = v/2, rate = 1/2 where v is the degrees of freedom. It is the distribution of the sum of squared Gaussian random variables
* inverse Gamma distribution 
39. The Empirical distribution involves using a set of delta functions to approximate the pdf. The cdf is approximated with a series of step functions
40. A discrete convolution operation involves flipping y, dragging it along and performing element-wise multiplication with x 
41. Central limit theorem states that the distribution of the sum of n random variables of any distribution converges to the standard normal
42. Monte Carlo approximation: approximate p(y) by drawing many samples from p(x) and applying f(x) on samples
43. Epistemic uncertainty: also known as model uncertainty
44. Aleotoric uncertainty: also known as data uncertainty, irreducible
45. Random variables can be constant such as the indicator function
46. The probability density function is the derivative of the cumulative density function
47. The inverse of the cdf is called the inverse cdf/percent point function (ppt)/quantile function
48. Unconditional independence means p(X,Y) = p(X)p(Y), while conditional independence means p(X,Y|Z) = p(X|Z)p(Y|Z)
49. Linearity of expectation: E[aX+b] = aE[x]+b
50. V[aX+b] = a^2V[X]
51. Law of iterated expectations/law of total expectation: E[X]=EY[E[X|Y]] i.e. take weighted average of subpopulations
52. Law of total variance/conditional variance formula: V[X]=EY[V[X|Y]]+VY[E[X|Y]] i.e. the overall variance of a random variable X can be evaluated as the sum of the within-sample and between-sample of X sampled on another random variable Y.
53. Bayes' rule is prior distribution (p(H=h)) multiplied by the likelihood (p(Y=y|H=h)) normalised by the marginal likelihood (p(Y=y))
54. The likelihood is a function since y is fixed, while it is called the observation distribution if y not fixed
55. Inverse probability: inferring the state of the world from observations of outcomes. Bayes' theorem solves the inverse probability problem (posterior distribution)

## Chapter 3
1. The covariance measures the degree to which two variables are linearly related
2. The covariance matrix applies to a single vector, while cross-covariance applies to two vectors
3. The correlation coefficient measures the degree of linearity, not the slope of the relationship
4. While independent implies uncorrelated, uncorrelated does not imply independent because correlation measures linear dependence
5. Simpson’s paradox states that a statistical trend that appears in several different groups of data can disappear or reverse when combined
6. The multivariate Gaussian is the most common joint probability distribution used for continuous random variables
7. There are different types of covariance matrices: full (elliptical contour), diagonal (axis-aligned ellipse) and spherical (circular shape), which have different number of free parameters
8. Mahalanobis distance measures the distance between a point and a distribution. It works by transforming the columns into uncorrelated variables, scaling columns to make variance equal to 1, and finally calculating the Euclidean distance
9. In terms of level sets, contours of constant (log) probability are equivalent to contours of constant Mahalanobis distance
10. The key property of an exponential family is that theta (parameters) and x (inputs) interact only via the dot product
11. The first and second cumulant are equal to the mean and variance
12. For the exponential family, higher order cumulants can be generated as derivatives of the log partition function
13. A mixture model involves creating more complex probability models by taking a convex combination of simple distributions
14. Gaussian mixture models are used in unsupervised clustering 
15. Convex combination - all coefficients are non-negative and sum to 1
16. A probabilistic graphical model is a joint probability distribution that uses a graph structure to encode conditional independence assumptions
17. A Bayesian network refers to a probabilistic graphical model which uses a directed acyclic graph structure
18. Each node represents a random variable, and each edge represents a direct dependency. Nodes not connected by edges are conditionally independent
19. The ordered Markov property involves connecting a DAG such that each node is conditionally independent of all its predecessors given its parents
20. First order Markov condition: the future is conditionally independent on the past given the present
21. The first order Markov condition is an example of parameter tying, because the state transition matrix is assumed to apply for all time steps (homogenous/stationary/time-invariant) and therefore can model an arbitrary number of variables using a fixed number of parameters
22. First-order Markov condition can be generalised to M’th order Markov models, by dependence on the M number of previous steps. 
23. Berkson’s paradox: negative mutual interaction between multiple causes of observations, also known as the explaining away effect
24. Inference in statistics refers to quantifying uncertainty about an unknown quantity estimated from a finite sample of data (different to deep learning use of inference for prediction)
25. Maximum likelihood estimation (MLE): pick parameters that assign the highest probability to the training data i.e. argmax of the probability of observing the data given parameters
26. Justification for MLE
* MLE is a special case of MAP where the prior is a uniform distribution
* The resulting predictive distribution is as close as possible to the empirical distribution of the data
27. The Kullback-Leibler divergence measures the similarity between probability distributions p and q. The '||' have no special meaning, except to emphasise that the order is important
28. The method of moments (MOM) is a simpler alternative to MLE, which equate the theoretical moments of the distribution to the empirical moments, and solve the resulting simultaneous equations
29. MOM is theoretically inferior to MLE because it may not use all the data as efficiently, and can sometimes produce inconsistent results
30. Empirical risk minimisation: the term empirical means we minimise our error based on a sample from the whole input domain, in contrast to the true error which uses the whole input domain
31. Overfitting is where the empirical risk is minimised but the true error increases
32. ERM is useful because we can produce an upper bound on the error
33. Exponentially weighted moving average has a bias because the estimate starts at mean = 0. This can be corrected by scaling by mean / 1 - beta^
34.  Regularisation adds a complexity penalty weighted by the regularisation parameter lambda
35. The inverse Wishart distribution is a probability distribution defined on real-valued positive-definite matrices. It is used as the conjugate prior for the covariance matrix of a multivariate normal distribution
36. One standard error rule: often we pick the model with the lowest CV error, but does not take into account uncertainty. Uncertainty involves taking the standard error of the out-of-sample error (mean of k-folds). The rule involves choosing the model with the lowest complexity within one standard error of the best-performing model’s CV
37. Bayes error is the error of the optimal predictor (the true model) due to inherent noise
38. Bayes model averaging: uses predictions from all possible models weighted by their posterior distribution
39. Conjugate priors: a prior for a likelihood function if the posterior is in the same parameterised family as the prior
40. Laplace’s rule of succession: rather than viewing p = s / n where s is the number of successes, use a more conservative estimate of p = n+1/n+2
41. Posterior predictive distribution: the distribution of future data given a posterior distribution
42. Marginal likelihood/evidence: useful for model comparisons, considering the evidence for the model across all possible parameter settings
43. Mixture of conjugate priors: can be used to better approximate the actual non-conjugate prior than a single conjugate prior, yet allows for a closed form approximation to the posterior
44. The Dirichlet distribution is a multivariate generalisation of the beta distribution
45. Hierarchical prior: prior for a prior i.e. a hyperprior/hyperparameter
46. Empirical Bayes: two step process where you first estimate the overall distribution of the data, and then use that distribution as a prior for estimating each average
47. Credible interval: Bayesian setting of confidence intervals, where parameter is not fixed. It is an interval such that the probability that the parameter lies in the interval is at least 1 - alpha. Requires using the inverse cdf, which can be hard for non-conjugate posteriors - instead using MC sampling, rank samples and then select samples at positions  alpha / S where S is the number of samples
48. Central interval is where the credible interval is centred on the estimate. A problem with central intervals is that there might be points with a higher probability outside the central interval 
49. Highest posterior density: to overcome the above problem, take a horizontal line and shift it up until the area above it but under the density is 1 - alpha
50. Plugin estimate: estimate a function g(theta) by using g(theta-hat) where theta-hat is an estimate of theta
