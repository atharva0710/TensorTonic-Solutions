import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.array(X, dtype = float)
    Y = np.array(y, dtype = float)

    N, D = X.shape

    w = np.zeros(D, dtype = float)
    b = 0.0

    for i in range(steps):
        z = np.matmul(X, w) + b
        p = _sigmoid(z)

        error = p - y
        grad_w = (X.T @ error) / N   # shape (D,)
        grad_b = np.mean(error)      # scalar

        # Gradient descent parameter update
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b
