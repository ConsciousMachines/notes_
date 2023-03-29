# this is a cheap "batched" version of the online arbitrary depth NN. 
# here I just keep a list of gradients for each element of the batch, 
# sum them at the end. 
# later I looked at what Nielsen did and his solution was the same thing!

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
num_inp = 2
eta = -1.
batch_size = 4


# init weights 
np.random.seed(0)
# layer                   0 1 2 3
hidden_structure = [num_inp,5,3,1]
W = [None] # so that weights of layer i will be at index i
B = [None]
for i in range(1, len(hidden_structure)):
    W.append(np.random.uniform(size=[hidden_structure[i], hidden_structure[i-1]]))
    B.append(np.random.uniform(size=[hidden_structure[i], 1]))


losses = []
for ii in range(2500):

    # forward pass 
    input = X.T
    y_s = [None]
    h_s = [X.T]
    for j in range(1, len(hidden_structure)):
        z = W[j] @ input + B[j]        # weighted input
        h = sigmoid(z)                 # hidden output
        input = h                      # propagate this to next layer's input 
        y_s.append(z)
        h_s.append(h)


    # calculate cost
    error = input - Y.T
    losses.append(np.sum(error ** 2))


    # we also have 1 less jacobian when dealing with ys rather than hs because x can count as an h (output of prev layer)
    jacobians = [] # each entry is grad_Yi_wrt_Yiminus1
    for j in range(2, len(hidden_structure)): 
        #jacobians.append(W[j].T * sigmoidp(y_s[j-1])) # grad_Yi_wrt_Yiminus1
        batch_jacobs = []
        _l = y_s[j-1].shape[0]
        for k in range(batch_size):
            batch_jacobs.append(W[j].T * sigmoidp(y_s[j-1])[:,k].reshape([_l,1]))
        jacobians.append(batch_jacobs)



    cumul_jacob = [[None for j in range(batch_size)] for i in range(len(jacobians) + 1)]
    cumul_jacob[-1] = np.ones([batch_size,1]) # the last is jacobian grad_H4_wrt_H4
    for j in range(-1,-len(jacobians)-1,-1):
        for k in range(batch_size):
            cumul_jacob[j-1][k] = jacobians[j][k] @ cumul_jacob[j][k]


    # update weights 
    common = eta * error * sigmoidp(y_s[-1]) # batch_size scalars
    for j in range(1, len(hidden_structure)):
        # we are supposed to be multiplying 5x1 times 1x2 here 
        for k in range(batch_size):
            B[j] += common[0][k] * cumul_jacob[j-1][k].reshape(B[j].shape)
            W[j] += common[0][k] * cumul_jacob[j-1][k].reshape(B[j].shape) @ h_s[j-1][:,k].reshape([1,W[j].shape[1]])


plt.plot(losses)
plt.ylim(0, 2)
plt.show()

