
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
10. Independence variables: P(A|B) = P(A). 
11. Linearity of expectation: E[aX+b] = aE[x]+b
12. V[aX+b] = a^2V[X]
13. Law of iterated expectations/law of total expectation: E[X]=EY[E[X|Y]] i.e. take weighted average of subpopulations
14. Law of total variance/conditional variance formula: V[X]=EY[V[X|Y]]+VY[E[X|Y]] i.e. the overall variance of a random variable X can be evaluated as the sum of the within-sample and between-sample of X sampled on another random variable Y.
15. Bayes' rule is prior distribution (p(H=h)) multiplied by the likelihood (p(Y=y|H=h)) normalised by the marginal likelihood (p(Y=y))
16. The likelihood is a function since y is fixed, while it is called the observation distribution if y not fixed
17. Inverse probability: inferring the state of the world from observations of outcomes. Bayes' theorem solves the inverse probability problem (posterior distribution)
18. The binomial distribution is a generalisation of the Bernoulli distribution for multiple events. The binomial coefficient is the formula for calculating combinations (where order does not matter, in contrast to permutations)
19. The logit function is the inverse of the sigmoid function, with domain [0,1]
20. The multinomial distribution generalises the binomial distribution to more than two outcomes, and generalises the categorical distribution to more than one event
21. The softmax function is the multinomial logit
22. The term 'temperature' related to the softmax function comes from the Boltzmann distribution which has the same form as the softmax function
23. Log-sum-exp trick: numerical stability trick to avoid exponentiating large numbers in the softmax formula, and applied to the cross entropy loss by modifying it to accept logits rather than probabilities. LSE works because we can shift the values in the exponent by an arbitrary constant c while still computing the same final value, and if we set c = max{x1...xn} then the largest positively exponentiated term is exp(0) = 1
24. The precision of a Gaussian is the inverse of the variance
25. The probit function is the inverse of the cdf of a Gaussian
26. Homoscedastic regression: variance is fixed and independent of input. Opposite of heteroscedastic regression
27. The softplus ensures that the resultant output is non-negative
28. The Gaussian distribution is the most popular distribution in statistics because:
* two easy to interpret parameters
* Central limit theorem: good for modelling residual errors i.e. noise
* makes the least number of assumptions (i.e. maximum entropy) subject to the constraint of a specified mean and variance
29. The Dirac delta function is what happens if the variance of the Gaussian approaches 0, making an infinitely tall spike
30. The sifting property of the Dirac delta function is important: Dirac delta function shifted by an amount -x. Multiplying the Dirac delta function by f(y) and integrating, gives the value of f(y) at the x
31. The Student t-distribution is an alternative to the Gaussian distribution that is robust to outliers. This is because the probability density decays as a polynomial function of the squared distance from the centre, compared to the exponential function in a Gaussian distribution, so there is more probability mass (heavy tails) at the ends. v is the number of degrees of freedom, where higher v makes it approach a Gaussian distribution
32. The Cauchy/Lorentz distribution is the Student distribution with v = 1. This has very heavy tails compared to the Gaussian
33. The half Cauchy distribution is folded over on itself so all its probability density are positive real values
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
