# from https://towardsdatascience.com/how-neural-networks-solve-the-xor-problem-59763136bdd7

# after I wrote the gradient descent equations on paper, and got an ugly but working result, 
# I looked online at how other people did it. This article was the best. I pulled up my code
# side by side with this one and renamed variables and re-arranged everything so that they 
# were near identical, to see the differences. In the end the only actual difference was 
# this guy used elegant linear algebra in the "calculate gradient" part while I had a wonky
# 3D stack of derivatives. 
# Going forward I ended up making an arbitrary depth dense network which used einsum to 
# merge a few operations, rather than continue transposing and matmul'ting. 

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
X = np.array([[0,0,1,1],[0,1,0,1]]).T
Y = np.array([[0,1,1,0]]).T

# initialize weights
eta = 0.2
num_input = 2
num_hidden = 2
num_output = 1
np.random.seed(0)
W1 = np.random.uniform(size=(num_input, num_hidden))
W2 = np.random.uniform(size=(num_hidden, num_output))
B1 = np.random.uniform(size=(1,num_hidden))
B2 = np.random.uniform(size=(1,num_output))


losses = []
for i in range(8000):

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