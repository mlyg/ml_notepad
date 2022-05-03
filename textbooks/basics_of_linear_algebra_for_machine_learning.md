# Basics of Linear Algebra for Machine Learning

## Chapter 1
1. **Numerical linear algebra** is another name for **applied algebra**

## Chapter 2

## Chapter 3

## Chapter 4

## Chapter 5
1. **numpy.reshape** can be used to add dimensions

## Chapter 6
1. **Broadcasting** can only be performed if the shape of each dimension in the arrays are equal or one has a dimension size of 1

## Chapter 7

## Chapter 8
1. **Max norm** (L inf) returns the maximum value of the vector

## Chapter 9

## Chapter 10
1. An **orthogonal** matrix is denoted by **Q**
2. Multiplication by an orthogonal matrix preserves length
3. A matrix is orthogonal if its transpose is equal to its inverse
4. Orthogonal matrices are used for **reflections** and **permutations**

## Chapter 11
1. The **determinant** gives the volume of a box with sides given by the rows of the matrix
2. **Rank** refers to the number of linearly independent rows or columns of a matrix

## Chapter 12
1. **Sparsity** is the number of non-zero elements divided by the total number of elements
2. Sparse matrices are used in encoding schemes: **one-hot encoding**, **count encoding** (word frequency) and **TF-IDF** (normalised word frequency)
3. Data structures to efficiently construct a sparse matrix:
* **Dictionary of keys**: row and column index mapped to a value
* **List of lists**: each row of the matrix is stored as a list, with each sublist containing a column index and value
* **Coordinate list**: list of tuples (row index, column index, value)
4. Data structures to perform efficient operations on sparse matrices:
* **Compressed Sparse Row**: three one-dimensional arrays for non-zero values, extent of rows and column indices
* **Compressed Sparse Column**: same as Compressed Sparse Row except column indices are compressed and read first before row indices

## Chapter 13
1. **Tensor product**: multiplying tensor A with q dimensions with tensor B with r dimensions, produces tensor with q + r dimensions
2. There are several types of tensor multiplication: **tensor product**, **tensor dot product** and **tensor contraction**

## Chapter 14
1. **LU decomposition**: A = LU, where A is a **square matrix**, L is a **lower triangle matrix** and U is an **upper triangle matrix**
* A more numerically stable version is **A = LUP**, where the parent matrix is re-ordered, and P specifies the way to return the result to the original order
* Used for solving systems of linear equations such as finding coefficients for linear regression or calculating determinant or inverse of a matrix
* Imported as: **from scipy.linalg import lu**
2. **QR decomposition**: A = QR, where A is an **nxm matrix**, Q is an **nxm matrix**, and R is an **upper triangle matrix**
* While LU is only for square matrices, QR can be used for nxm matrices
* Imported as: **from numpy.linalg import qr**
* Returns Q and R matrices with reduced dimensions, can return full nxm matrix with **mode = 'complete'**
3. **Cholesky decomposition**: A = LL^T, where A is a **square, symmetrical matrix** with all values **greater than 0**, L is a **lower triangular matrix** 
* Can also be written as A = UU^T, where U is an upper triangular matrix
* Used for solving linear least squares for linear regression, as well as simulation and optimisation methods
* More than **twice as efficient as LU**
* Import as: **from numpy.linalg import cholesky**
* Function only returns L (can get L^T easily)

## Chapter 15
1. **Eigendecomposition** refers to decomposing a square matrix into eigenvectors and eigenvalues
2. Either **Av = lv** or **A = QLQ^T**, where Q is a matrix of eigenvectors, L is the diagonal matrix of eigenvalues and Q^T is the transpose of the matrix of eigenvectors
3. Eigenvectors have unit magnitude, and are typically the **right vectors** (column vector)
4. Eigenvalues scale eigenvectors

## Chapter 16
1. All matrices have an **SVD**, but not all have an eigendecomposition
2. For a real value matrix, A = U S V^T, where A is a **real nxm matrix**, U is a **mxm matrix**, S is a mxm **diagonal matrix** and V^T is an **nxn matrix**
3. The diagonal values of the Sigma (S) matrix are the **singular values** of the original matrix
4. The **columns** of U are the **left-singular** vectors of A
5. The **columns** of V are the **right-singular** vectors of A
6. SVD is used to get a **matrix inverse**, **decompressing data**, **least squares linear regression**, **image compression**, and **denoising data**
7. With SVD using numpy, to reconstruct the matrix the s vector must be converted with diag(). However, this will return an mxm matrix, and if the original is mxn, you first need to create a zero mxn array, and then populate it with the diag(s)
8. The **Moore-Penrose** (pseudoinverse) inverse generalises matrix inverse from square to rectangular matrices
9. It is denoted as A+, and calculated by A+ = V D+ U^T, where D+ is the pseudoinverse of the diagonal matrix S, and V and U are from the SVD
10. D+ can be calculated by creating a **diagonal matrix S**, calculating the **reciprocal** of all non-zero elements and then taking the **transpose** if the original matrix is rectangular
11. SVD can perform **dimensionality reduction** by selecting the **top k largest singular values** in S (columns from S, and rows from V^T)

## Chapter 17
1. For the **variance**, the denominator subtracts 1 to **correct for the bias** (related to degrees of freedom)
2. numpy.var() calculates population variance, must set **ddof=1** for the sample variance (where the denominator subtracts 1)
3. numpy.std() calculates population standard deviation, and again must set ddof=1 for sample standard deviation
4. **Covariance** measures the **joint probability of two random variables**, and is calculated as the expected value of the product of the differences of each random variable from their expected values
5. **Covariance sign** shows whether two variables increase together (positive) or decrease together (negative). Covariance of zero means variables are **independent**. The **magnitude is not easily interpreted**
6. **Pearson correlation coefficient**: covariance **normalised** by dividing by the product of standard deviations, giving score between -1 and 1
7. The **covariance** matrix is **square** and **symmetrical**, where the diagonal of the covariance matrix are the variances of the random variables
8. **numpy.cov()** requires the data to be in the format where the **features are columns**

## Chapter 18
1. PCA is a **projection method** where data with m-columns (features) is projected into a subspace with m or fewer columns
2. The covariance method of calculating PCA involves **standardising** the data, getting the **covariance matrix**, performing **eigendecomposition** on the covariance matrix and then sorting the top k eigenvalues to get a matrix containing the **top k eigenvectors**

## Chapter 19
1. **Linear regression** models the relationship between two scalar values, assuming y is a linear combination of the input variable x
2. The objective of creating a linear regression model is to find values for the coefficient values (b) that minimise the error in prediction of the output variable y
3. **Linear least squares** involves finding the line that **minimises the squared error**
4. The regression problem can be solved directly by **matrix inverse**, using the **normal equation**: b = (X^T X)^-1 X^T y
5. However, matrix inverse is both computationally expensive and numerically unstable
6. The **QR decomposition** is a popular method to solve linear least squares equation: b = R^-1 Q^T y
7. There is still matrix inversion, but on the simpler R matrix
8. While more numerically stable and computationally efficient, it does not work for all matrices
9. **SVD** is the more stable and preferred approach: b = X+ y where X+ = U D+ V^T
10. Numpy provides the l**stsq()** function which uses the **SVD** approach
