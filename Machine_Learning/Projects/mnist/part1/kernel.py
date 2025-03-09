import numpy as np

### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    return (np.dot(X, Y.T) + c)**p
    # YOUR CODE HERE
    raise NotImplementedError



def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    #FORMULA
    # norm(x-y)^2 = norm(x^2) + norm(y^2) - 2*(x*y)

    X_sq = np.sum(X**2, axis=1, keepdims=True)  # (n, 1)
    Y_sq = np.sum(Y**2, axis=1, keepdims=True)  # (m, 1)
    
    dist_sq = X_sq - 2 * np.dot(X, Y.T) + Y_sq.T  # (n, m)
    
    # Aplicar la función exponencial con el parámetro gamma
    return np.exp(-gamma * dist_sq)
    # YOUR CODE HERE
    raise NotImplementedError
