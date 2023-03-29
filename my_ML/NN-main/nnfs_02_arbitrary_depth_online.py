# after seeing that Nielsen used a list to describe a NN Architecture, I tried it.
# now we can define an NN by a list of hidden units at that layer. 
# all we do is: 
# 1. figure out the Jacobian between layers (activation at l+1 wrt activation at l)
# 2. multiply them together to get a cumulative Jacobian between farther layers 
# 3. iterate over the architecture list to get jacobians at each layer. 
# this was inspired by colah and nielsen. 

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
eta = 1.

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
    y = Y[ii % 4].reshape([1,1]) 
    # intermediate stuff 
    input = x
    y_s = [None]
    h_s = [x]
    for j in range(1, len(hidden_structure)):
        z = W[j] @ input + B[j]        
        h = sigmoid(z)      
        input = h           
        y_s.append(z)
        h_s.append(h)


    # calculate cost
    error = input - y
    losses.append(np.sum(error ** 2))

    # calculate gradient, sum over paths in the graph
    jacobians = [None] 
    for j in range(1, len(hidden_structure)):
        jacobians.append(sigmoidp(y_s[j]) * W[j])

    cumul_jacob = [None for i in range(len(jacobians))]
    cumul_jacob[-1] = np.ones([1,1]) 
    cumul_jacob[-2] = jacobians[-1] 
    for i in range(-2, -len(jacobians), -1):
        cumul_jacob[i-1] = cumul_jacob[i] @ jacobians[i] 


    # update weights 
    for i in range(1, len(hidden_structure)):
        common = eta * error * cumul_jacob[i].T * sigmoidp(W[i] @ h_s[i-1] + B[i])
        W[i] -= common @ h_s[i-1].T 
        B[i] -= common
        
plt.plot(losses)
plt.ylim(0, 2)
plt.show()
