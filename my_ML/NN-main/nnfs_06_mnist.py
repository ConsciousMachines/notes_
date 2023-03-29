# the einsum calculation has been perfected here.
# taking my XOR NN and applying it to MNIST by simply changing some dimensions.

import pickle
import gzip
import numpy as np
import time

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoidp(x):
    s = sigmoid(x)
    return s * (1 - s)

def sigmoidq(x):
    return x * (1 - x)

def one_hot_encode(j):
    e = np.zeros((10))
    e[j] = 1.0
    return e

# load data
f = gzip.open(r'C:\Users\pwnag\Desktop\sup\nielsen\mnist.pkl.gz', 'rb')
_tr, _va, _te = pickle.load(f, encoding = "latin1")
f.close()
tr_x = [np.reshape(x, (784, 1)) for x in _tr[0]] # reshape x's
va_x = [np.reshape(x, (784, 1)) for x in _va[0]]
te_x = [np.reshape(x, (784, 1)) for x in _te[0]]
tr_y = np.array([one_hot_encode(y) for y in _tr[1]]) # one-hot-encode the y's
tr_data = list(zip(tr_x, tr_y))
va_data = list(zip(va_x, _va[1])) # list of tuples of (x,y)
te_data = list(zip(te_x, _te[1]))

def feedforward(a):
    for _i in range(1, len(B)):
        a = sigmoid(W[_i] @ a + B[_i])
    return a

###############################################################################
###############################################################################




# params
batch_size = 10
eta = 1.0
np.random.seed(1)
num_inp = 784
num_out = 10
# layer                   0 1 2 3
hidden_structure = [num_inp,30,num_out]
B = [None] + [np.random.randn(y, 1) for y in hidden_structure[1:]]
W = [None] + [np.random.randn(y, x) for x, y in zip(hidden_structure[:-1], hidden_structure[1:])]


__id3 = np.zeros([batch_size, num_out, num_out])
__idx = np.arange(num_out)
__id3[:,__idx,__idx] = 1


for _e in range(30): # for each epoch

    s = time.time()

    __indices = np.random.permutation(50_000).reshape(50_000 // batch_size, batch_size)

    for _poop in range(50_000 // batch_size):

        # create batch
        __idx = __indices[_poop]
        batch_x = _tr[0][__idx,:].T
        batch_y = tr_y[__idx,:].T


        # forward pass 
        input = batch_x
        z_s = [None]
        h_s = [batch_x]
        for j in range(1, len(hidden_structure)):
            z = W[j] @ input + B[j]        
            input = sigmoid(z)                
            z_s.append(z)
            h_s.append(input)


        # get the jacobians and gradients
        jacobians = [np.einsum('ij,il->lij',W[j].T, sigmoidq(h_s[j-1])) for j in range(2, len(hidden_structure))]
        cumul_jacob = [None for i in jacobians] + [__id3] 
        for j in range(-1,-len(jacobians)-1,-1):
            cumul_jacob[j-1] = np.einsum('pij,pjk->pik', jacobians[j], cumul_jacob[j]) # matmul in inner 2 dimensions


        # update weights - comment out sigmoidq to get cross-entropy from squared loss lol
        error = input - batch_y
        common = (eta / batch_size) * error * sigmoidq(h_s[-1]) # dCost_dy, y is pre-activation in last layer
        for j in range(1, len(hidden_structure)):
            B[j] -= np.expand_dims(np.einsum('lj,jkl->k', common, cumul_jacob[j-1]), 1)
            W[j] -= np.einsum('ij,jhi,wj->hw', common, cumul_jacob[j-1], h_s[j-1])

    # test on te_data
    __test_results = [(np.argmax(feedforward(x)), y) for (x, y) in te_data]
    __evaluate = sum(int(x == y) for (x, y) in __test_results)
    print(f"Epoch {_e} : {__evaluate} / {len(te_data)}\ttime: {time.time() - s}")

