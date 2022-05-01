# Hands-on genetic algorithms with python

## Chapter 1
1. Darwinian evolution theory involves three principles: principle of **variation**, **inheritance** and **selection**
3. **Holland's schema theorem**: known as the fundamental theorem of genetic algorithms, states that the frequency of schemata of **low order** (number of fixed digits), **short defining length** (largest distance between fixed digits), and above average fitness **increases exponentially** in successive generations
4. The key characteristics of genetic algorithms are: maintain a **population of solutions**, use a **genetic representation** of the solutions, use the outcome of a **fitness function** and is **probabilistic**
5. **Advantages** of genetic algorithms: 
* global optimisation
* can handle problems with complex/no mathematical representation
* robust to noise
* can be parallelised
* can be used for continual learning
7. **Disadvantages** of genetic algorithms
* need special definitions
* need hyperparameter tuning
* computationally-expensive
* risk of premature convergence
* no guaranteed solution

## Chapter 2
1. **Selection methods**
* **Roulette wheel selection/fitness proportionate selection**: probability of selecting individual proportional to fitness
* **Stochastic universal sampling**: uses multiple selection points so all individuals are chosen at the same time, providing weaker individuals with a chance to be chosen
* **Rank-based selection**: use fitness values to sort individuals, useful when few individuals have much larger fitness values than others, or if fitness values close together
* **Fitness scaling**: applies scaling directly to fitness values
* **Tournament selection**: select subset of population and choose individual with highest fitness out of subset, and repeat
2. **Crossover methods**
* **Single-point crossover**: swap genes to right of crossover point
* **Two-point crossover**: swap genes in-between two points
* **Uniform crossover**: genes are passed randomly between parents
* **Ordered crossover**: preserves relative order of parent genes
* **Blend crossover**: offspring takes values in interval between (or beyond) parents
* **Simulated binary crossover**: average of parents value is equal to offspring's value
3. **Mutation methods**
* **Flip bit mutation:** one gene randomly selected and value flipped
* **Swap mutation**: two genes selected and values swapped
* **Inversion mutation**: order of genes in sequence reversed
* **Scramble mutation**: random sequenceo f genes selected and order of genes randomised
* **Gaussian mutation**: apply Gaussian noise
4. **Elitism**: guarantees best individuals make it to next generation
5. **Niching and sharing**: used to find several optimal solutions not just global optimum. Can be done either with serial niching or parallel niching.
