# Here is an elegant linear algebra solution to the batched equations. 
# I used a matrix calculus result from:
# https://mostafa-samir.github.io/auto-diff-pt2/
# to get the gradient in terms of df_dY instead of using grad_Hi_wrt_Himinus1 like last time
# the code was broken for a bit because i didnt realize i had both y as the label output and y = Wx + b -.-

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

# init weights 
np.random.seed(0)
# layer                   0 1 2 3
hidden_structure = [num_inp,2,1]
W = [None] # so that weights of layer i will be at index i
B = [None]
for i in range(1, len(hidden_structure)):
    W.append(np.random.uniform(size=[hidden_structure[i], hidden_structure[i-1]]))
    B.append(np.random.uniform(size=[hidden_structure[i], 1]))


losses = []
for ii in range(10_000):

    # forward pass 
    x = X[ii % 4].reshape([2,1]) 
    yyy = Y[ii % 4].reshape([1,1]) 
    # intermediate stuff 
    input = x
    y_s = [None]
    h_s = [x]
    for j in range(1, len(hidden_structure)):
        z = W[j] @ input + B[j]        # weighted input
        h = sigmoid(z)                 # hidden output
        input = h                      # propagate this to next layer's input 
        y_s.append(z)
        h_s.append(h)


    # calculate cost
    error = input - yyy
    losses.append(np.sum(error ** 2))


    # we also have 1 less jacobian when dealing with ys rather than hs because x can count as an h (output of prev layer)
    jacobians = [] # each entry is grad_Yi_wrt_Yiminus1
    for j in range(2, len(hidden_structure)): 
        jacobians.append(W[j].T * sigmoidp(y_s[j-1])) # grad_Yi_wrt_Yiminus1


    cumul_jacob = [None for i in range(len(jacobians) + 1)]
    cumul_jacob[-1] = np.ones([1,1]) # the last is jacobian grad_H4_wrt_H4
    for j in range(-1,-len(jacobians)-1,-1):
        cumul_jacob[j-1] = jacobians[j] @ cumul_jacob[j]


    # update weights 
    common = eta * error * sigmoidp(y_s[-1])
    for j in range(1, len(hidden_structure)):
        W[j] += common * cumul_jacob[j-1] @ h_s[j-1].T
        B[j] += common * cumul_jacob[j-1]


plt.plot(losses)
plt.ylim(0, 2)
plt.show()

