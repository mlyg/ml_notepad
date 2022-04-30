# Basics of Linear Algebra for Machine Learning

## Chapter 1
1. Numerical linear algebra is another name for applied algebra

## Chapter 2

## Chapter 3

## Chapter 4

## Chapter 5
1. numpy.reshape can be used to add dimensions

## Chapter 6
1. Broadcasting can only be performed if the shape of each dimension in the arrays are equal or one has a dimension size of 1

## Chapter 7

## Chapter 8
1. Max norm (L inf) returns the maximum value of the vector

## Chapter 9

## Chapter 10
1. An orthogonal matrix is denoted by Q
2. Multiplication by an orthogonal matrix preserves length
3. A matrix is orthogonal if its transpose is equal to its inverse
4. Orthogonal matrices are used for reflections and permutations

## Chapter 11
1. The determinant gives the volume of a box with sides given by the rows of the matrix
2. Rank refers to the number of linearly independent rows or columns of a matrix

## Chapter 12
1. Sparsity is the number of non-zero elements divided by the total number of elements
2. Sparse matrices are used in encoding schemes: one-hot encoding, count encoding (word frequency) and TF-IDF (normalised word frequency)
3. Data structures to efficiently construct a sparse matrix:
* Dictionary of keys: row and column index mapped to a value
* List of lists: each row of the matrix is stored as a list, with each sublist containing a column index and value
* Coordinate list: list of tuples (row index, column index, value)
4. Data structures to perform efficient operations on sparse matrices:
* Compressed Sparse Row: three one-dimensional arrays for non-zero values, extent of rows and column indices
* Compressed Sparse Column: same as Compressed Sparse Row except column indices are compressed and read first before row indices
