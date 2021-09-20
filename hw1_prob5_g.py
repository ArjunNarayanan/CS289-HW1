import numpy as np
from scipy.io import loadmat


def assemble_feature(x, D):
    '''
    x should be an Nx5 dimensional numpy array, where N is the number of data points
    D is the maximum degree of the multivariate polynomial
    '''
    n_feature = x.shape[1]
    Q = [(np.ones(x.shape[0]), 0, 0)]
    i = 0
    while Q[i][1] < D:
        cx, degree, last_index = Q[i]
        for j in range(last_index, n_feature):
            Q.append((cx * x[:, j], degree + 1, j))
        i += 1
    return np.column_stack([q[0] for q in Q])


def solve_ridge_regression(X, Y, lamda):
    ndofs = X.shape[1]
    op = X.T @ X + lamda * np.eye(ndofs)
    rhs = X.T @ Y
    return np.linalg.solve(op, rhs)


def cross_validation_train_test(X, Y, indices, testidx):
    trainindices = np.delete(indices, testidx)
    Xtrain = np.vstack(X[trainindices])
    Ytrain = np.concatenate(Y[trainindices])
    Xtest = X[testidx]
    Ytest = Y[testidx]
    return Xtrain, Ytrain, Xtest, Ytest


def cross_validation_error(splitx, splity, numfolds, D, lamda):
    error = np.zeros(numfolds)
    cvindices = np.array(range(numfolds))

    for idx in cvindices:
        Xtrain, Ytrain, Xtest, Ytest = cross_validation_train_test(splitx, splity, cvindices, idx)

        trainfeatures = assemble_feature(Xtrain,D)
        numdiag = trainfeatures.shape[1]
        op = trainfeatures.T @ trainfeatures + lamda*np.eye(numdiag)
        rhs = trainfeatures.T @ Ytrain

        sol = np.linalg.solve(op,rhs)

        testfeatures = assemble_feature(Xtest,D)
        predictions = testfeatures @ sol
        error[idx] = ((np.linalg.norm(predictions - Ytest))**2)/len(Ytest)

    return np.mean(error)


numfolds = 4
lamda = 0.1
polyorders = range(6)

data = loadmat("polynomial_regression_samples.mat")
x = data["x"]
y = np.ravel(data["y"])

npts = y.shape[0]
np.random.seed(42)
shuffleidx = np.random.permutation(range(npts))

xshuffle = x[shuffleidx, :]
yshuffle = y[shuffleidx]

numsplit = int(npts / numfolds)
splitpoints = np.linspace(0, npts, numfolds + 1, dtype=int)

splitx = np.array([xshuffle[splitpoints[i]:splitpoints[i + 1], :] for i in range(len(splitpoints) - 1)])
splity = np.array([yshuffle[splitpoints[i]:splitpoints[i + 1]] for i in range(len(splitpoints) - 1)])

err = [cross_validation_error(splitx,splity,numfolds,p,lamda) for p in polyorders]