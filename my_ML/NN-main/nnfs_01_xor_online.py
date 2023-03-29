# my first network, elegant equations after comparing to article. 
# ONLINE VERSION - as opposed to batch with entire 4 observations. 
# in retrospect I realize there is nothing different "in the middle"
# in online mode, out input is a 1xN vector x. 
# in batch mode, we have a matrix. so each column of the matrix
#     is being treated as an individual observation, and 
#     the computations are independent and separate. 
# so the only thing that's different is the very first matrix
#     multiplication of the input, and then we sum over an axis 
#     in the end after dotting with error properly. 


import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidp(x):
    s = sigmoid(x)
    return s * (1 - s)

def sigmoidq(x):
    return x * (1 - x)

# init data
_X = np.array([[0,0,1,1],[0,1,0,1]]).T
_Y = np.array([[0,1,1,0]]).T

# initialize weights
eta = 0.2
num_input = 2
num_hidden = 2
num_output = 1
np.random.seed(1)
W1 = 2 * np.random.uniform(size=(num_input, num_hidden)) - 1
W2 = 2 * np.random.uniform(size=(num_hidden, num_output)) - 1
B1 = 2 * np.random.uniform(size=(1,num_hidden)) - 1
B2 = 2 * np.random.uniform(size=(1,num_output)) - 1

losses = []
for i in range(2*8000):
    ii = i % 4
    X = _X[ii].reshape([1,2])
    Y = _Y[ii].reshape([1,1])

    # forward pass 
    H = sigmoid(X @ W1 + B1)
    out = sigmoid(H @ W2 + B2)

    # calculate cost
    error = Y - out
    losses.append(np.sum(error ** 2))

    # calculate gradient
    common2 = eta * error * sigmoidp(H @ W2 + B2)
    common1 = common2 @ W2.T * sigmoidp(X @ W1 + B1)

    # update
    W2 += H.T @ common2
    B2 += np.sum(common2, axis = 0)
    W1 += X.T @ common1
    B1 += np.sum(common1, axis = 0)
    if i % 500 == 0:
        print("error: ", losses[i]) 
plt.plot(losses)
plt.ylim(0, 2)
plt.show()
losses[-20:]