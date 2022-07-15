# Chapter 1 Notes

# Chapter 1 Questions
1. The various layers of a neural network are the input, hidden layers and output
2. The output of feedforward propagation is from multiplying the inputs by weights and adding the bias term, passing through activation functions, and repeating this process through the neural network
3. Loss function of a continuous dependent variable e.g. MSE/MAE while for a categorical variable e.g. cross entropy loss
4. Stochastic gradient descent involves updating weights in the network using random samples from the dataset
5. Backpropagation involves updating weights towards their optimal value using partial derivatives to evaluate their contribution towards the loss function
6. Weight update involves calculating the partial derivatives of the loss with respect to all weights
7. Forward propagation and backpropagation happen within each epoch of training
8. GPU training is faster because it can be parallelised
9. Learning rate impacts the speed and stability of training
10. Typical value of a learning rate is between 0.0001 and 0.01

# Chapter 2 Notes
1. Torch tensors can only have one datatype, and all values are coerced into the most generic format
2. Torch max function returns both the maximum and the argmax
3. Always use permute rather than view to reshape a tensor 
4. torch_summary is a package that is useful for printing model summary
5. mode.state_dict() returns an ordered dictionary of parameter names (keys) and weights and biases (values)
6. Good practice to do torch.save(model.to('cpu').state_dict(), mymodel.pth)
7. To load model: model.load_state_dict(state_dict)

# Chapter 2 Questions
1. weights are computed as float values
2. A tensor object can be reshaped with view or permute, squeeze, unsqueeze
3. Computation is faster than with tensor objects than NumPy arrays because of GPU parallelisation
4. The init function involves defining the layers of the neural network
5. We zero gradients to prevent accumulation
6. The dataset class uses __getitem__ and __len__ magic functions
7. We make predictions on new data points by passing tensors into the neural network
8. Intermediate layer predictions can be obtained by using the forward method to output the intermediate values
9. Sequential method prevents the need to define the neural network as a class
