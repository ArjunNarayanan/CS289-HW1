import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def vandermonde_matrix(x, D):
    return np.column_stack([x ** i for i in range(D + 1)])


data = loadmat("1D_poly.mat")
x_train = np.ravel(data["x_train"])
y_train = np.ravel(data["y_train"])

op = vandermonde_matrix(x_train, 19)
sol = np.linalg.solve(op, y_train)
predictions = op @ sol

error = ((np.linalg.norm(predictions - y_train))**2)/20