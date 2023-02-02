import numpy as np
import numpy.matlib
import math
import matplotlib.pyplot as plt

def generate_sample(K, Theta):
    X = np.random.default_rng().normal(0, 2, (1, K))
    Epsilon = np.random.default_rng().normal(0, math.sqrt(0.1))
    Y = np.dot(Theta, X.transpose()) + Epsilon
    return np.append(X, Y, 1)

def generate_data(N, K):
    Theta = np.random.default_rng().standard_normal((1, K))
    Samples = np.empty((0, K+1))
    for _ in range(N):
        Sample = generate_sample(K, Theta)
        Samples = np.append(Samples, Sample, axis=0)
    return Theta, Samples

def calculate_squared_error(Array):
    SquaredDiff = np.square(Array)
    Loss = np.sum(SquaredDiff)
    return Loss

def calculate_loss(X, Theta, Y):
    return calculate_squared_error(np.matmul(X, Theta.transpose()) - Y)

def run_simulation(N, K, Lambda):
  Theta, Data = generate_data(N, K)
  X, Y = Data[:, :-1], Data[:, -1].reshape(N, 1)

  # if N is greater than or equal to K then solve OLS else use ridge regression
  if N >= K:
    ThetaStar = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose()), Y).reshape(1, K)
    # Loss = calculate_loss(X, ThetaStar, Y)
    return calculate_squared_error(Theta - ThetaStar)
  else:
    RidgeTheta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X) + Lambda ** 2 * np.matlib.identity(K)), X.transpose()), Y).reshape(1, K)
    # RidgeLoss = calculate_loss(X, RidgeTheta, Y)
    return calculate_squared_error(Theta - RidgeTheta)

KRange = [3, 10, 50, 100]
NRange = range(1, 101)
Lambda = 0.5

Results = []
for K in KRange:
  Temp = []
  for N in NRange:
    Temp.append(run_simulation(N, K, Lambda))
  Results.append(Temp)
Results = np.array(Results)

for i in range(len(KRange)):
  fig, ax = plt.subplots()
  fig.set_size_inches(25, 15)
  ax.plot(NRange, Results[i])
  ax.set_xlabel('N')
  ax.set_ylabel('L2 Distance between theta and estimated theta')
  plt.show()

Results = []
K = 1000
N = 50
LambdaRange = [x/1000 for x in range(1, 1001)]
for Lambda in LambdaRange:
  Results.append(run_simulation(N, K, Lambda))
Results = np.array(Results)

fig, ax = plt.subplots()
fig.set_size_inches(25, 15)
ax.set_xlabel('Lambda')
ax.set_ylabel('L2 Distance between theta and estimated theta')
ax.plot(LambdaRange, Results)
plt.show()

