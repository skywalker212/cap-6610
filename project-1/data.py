import numpy as np
import numpy.matlib
import math

def generate_datapoint(K, Theta):
    X = np.random.default_rng().normal(0, 2, (1, K))
    Epsilon = np.random.default_rng().normal(0, math.sqrt(0.1))
    Y = np.dot(Theta, X.transpose()) + Epsilon
    return np.append(X, Y, 1)

def generate_data(N, K):
    Theta = np.random.default_rng().standard_normal((1, K))
    Samples = np.empty((0, K+1))
    for _ in range(N):
        Sample = generate_datapoint(K, Theta)
        Samples = np.append(Samples, Sample, axis=0)
    return Theta, Samples

def calculate_squared_error(Array):
    SquaredDiff = np.square(Array)
    Loss = np.sum(SquaredDiff)
    return Loss

def calculate_loss(X, Theta, Y):
    return calculate_squared_error(np.matmul(X, Theta.transpose()) - Y)


K = 10
N = 3
Theta, Data = generate_data(N, K)
X, Y = Data[:, :-1], Data[:, -1].reshape(N, 1)

ThetaStar = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose()), Y).reshape(1, K)
Loss = calculate_loss(X, ThetaStar, Y)
Distance = calculate_squared_error(Theta - ThetaStar)
print(Loss, Distance)

Lambda = 0.5
RidgeTheta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X) + Lambda ** 2 * np.matlib.identity(K)), X.transpose()), Y).reshape(1, K)
RidgeLoss = calculate_loss(X, RidgeTheta, Y)
RidgeDistance = calculate_squared_error(Theta - RidgeTheta)
print(RidgeLoss, RidgeDistance)