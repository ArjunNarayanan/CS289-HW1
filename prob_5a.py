import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def vandermonde_matrix(x,D):
    return np.column_stack([x**i for i in range(D+1)])

def solve_least_squares(X,Y):
    op = X.T @ X
    rhs = X.T @ Y
    return np.linalg.solve(op,rhs)

def least_squares_error(X,Y,D):
    vandermonde = vandermonde_matrix(X,D)
    sol = solve_least_squares(vandermonde,Y)
    predictions = vandermonde @ sol
    npts = X.shape[0]
    error = ((np.linalg.norm(Y - predictions))**2)/npts
    return error


data = loadmat("1D_poly.mat")
x_train = np.ravel(data["x_train"])
y_train = np.ravel(data["y_train"])

Dmax = x_train.shape[0]
error = [least_squares_error(x_train,y_train,D) for D in range(Dmax)]

# fig,ax = plt.subplots()
# ax.plot(range(Dmax),error)
# ax.grid()
# ax.set_xlabel("D")
# ax.set_ylabel("Normalized error")
# fig.savefig("prob_5a.png")