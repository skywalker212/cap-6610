import numpy as np

def sig(x):
 return 1/(1 + np.exp(-x))

def calculate_label(XY, a, b, r):
    if (XY[0] - a)**2 + (XY[1] - b)**2 < r**2:
        return 1
    else:
        return 0
    

def generate_data(D, N, a, b, r):
    temp = np.random.default_rng().uniform(size=(D, N))
    Points = np.ones((D, N+1))
    Points[:, :-1] = temp
    Labels = np.array(list(map(lambda XY: calculate_label(XY, a, b, r), Points))).reshape((D, 1))
    return Points, Labels

D = 100
N = 2
eta = 0.5 
Threshold = 0.5
epochs = 1000

# Initialize weights
W_i = np.random.default_rng().uniform(0, 1, (11, 1))
W_ij = np.random.default_rng().uniform(0, 1, (10, N+1))

a = 0.5
b = 0.6
r = 0.4
# Generate the data
X_train, Y_train = generate_data(D, N, a, b, r)
X_test, Y_test = generate_data(D, N, a, b, r)

for _ in range(epochs):
    E = 0
    # SGD with batch size 1
    for i in range(D):
        x = X_train[i].reshape((N+1, 1))
        t = Y_train[i].item()
        
        # Output of hidden layer
        temp = np.dot(W_ij,x).reshape((10, 1))
        temp = np.array(list(map(sig, temp)), ndmin=2)
        H_i = np.append(temp, [[1]], axis=0)
        
        # Output of out layer with one node
        o = sig(np.dot(W_i.T, H_i).item())

        # Error
        E += ((t - o)**2)/2

        # Back prop
        delta_W_i = eta * (t - o) * o * (1 - o) * H_i
        delta_W_ij = eta * (t - o) * o * (1 - o) * np.dot((W_i * H_i * (1-H_i))[:-1, :], x.T)
        W_i += delta_W_i
        W_ij += delta_W_ij
    print("Avg Error after epoch: ", E/D)


# Forward prop for testing
temp = np.dot(X_test, W_ij.T)
temp = np.array(list(map(sig, temp)), ndmin=2)
H_di = np.ones((D, 11))
H_di[:, :-1] = temp

# Output converted to label 0/1 depending on the threshold set in Threshold variable (0.5)
O = np.array(list(map(lambda x: 1 if x >= Threshold else 0, sig(np.dot(H_di, W_i))))).reshape((D, 1))

# Counting number of mislabelings (if sum of true+predicted label is 1 then its a mislabelling and if it's 2 or 0 then its a correct labelling) and calculating the accuracy
Accuracy = (1 - (np.count_nonzero((O + Y_test) == 1)/D)) * 100
print(Accuracy)