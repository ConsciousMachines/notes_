# my original first network, with weird equations because I didn't figure out the linear algebra. 

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
eta = 0.2

# initialize weights
np.random.seed(0)
W1 = 2 * np.random.random(size=[2,2]) - 1
B1 = 2 * np.random.random(size=[1,2]) - 1
W2 = 2 * np.random.random(size=[2,1]) - 1
B2 = 2 * np.random.random(size=[1,1]) - 1



losses = []
for i in range(10_000):

    # forward pass 
    H = sigmoid(X @ W1 + B1)
    out = sigmoid(H @ W2 + B2)

    # calculate cost
    error = Y - out
    loss = np.sum(error*error)
    losses.append(loss)

    # derive f wrt its own weights 
    common2 = sigmoidp(H @ W2 + B2) # this is 4 dCost_dB2 for 4 separate instances... so i guess we add it?
    df_dB2 = np.sum(error * common2)


    # for the first of 2 weights, it's common2*h1 so multiply 4 instances of common2 times 4 instances of h1
    #df_dw31 = common2 * H[:,0].reshape([4,1])
    # likewise for the other weight
    #df_dw32 = common2 * H[:,1].reshape([4,1])
    # according to this experiment, multiplying 4x2 by 4x1 will multiply each column vector by the 4x1
    #H * np.array([[1,1,1,1],[0,0,0,0]]).T
    #H * np.array([[1,1,0,0]]).T
    _df_dW2 = H * common2
    #_df_dW2 == np.hstack([df_dw31, df_dw32])
    # then we sum over the 4 instances and result has same shape as W2 
    df_dW2 = np.sum(error * _df_dW2, axis=0).reshape([2,1])


    # this is a horizontal stack of the two 4x1 vectors df_dh1 and df_dh2
    #np.hstack([sigmoidp(H @ W2 + B2) * W2[0], sigmoidp(H @ W2 + B2) * W2[1]]) 
    df_dH = sigmoidp(H @ W2 + B2) @ W2.T
    common1 = df_dH * sigmoidp(X @ W1 + B1)
    dh_dB1 = np.sum(error * common1, axis=0).reshape([1,2])


    # one instance yields a 2x2 matrix for dW1:
    stak = np.zeros([4,2,2])
    for p in range(4):
        stak[p] = X[p].reshape([2,1]) @ common1[p].reshape([1,2])

    df_dW1 = np.sum(stak * error.reshape([4,1,1]), axis = 0)

    B2 += eta * df_dB2
    W2 += eta * df_dW2
    B1 += eta * dh_dB1
    W1 += eta * df_dW1

losses[-20:]
plt.plot(losses)
plt.ylim(0,max(losses))
plt.show()



