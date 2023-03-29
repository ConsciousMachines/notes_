# Ok I took apart Nielsen's NN. Could not care enough to
# take apart the nabla_B stuff, but after renaming all the 
# variables to the names I use, we ended up with the exact
# same W and B. so I guess I'm not wrong.

# Lesson learned: a NN that outputs 10 rather than 1 
# is the same as a NN that outputs 1, just there are 
# 10 outputs. lol. The only thing that had to change is
# the derivative in the last cumul_jacob. before it was 
# batch x 1 x 1, now it is a stack of batch_size identity
# matrices of size num_out x num_out. 
# also einsum had to change a bit to accomodate the new 
# dimension, rather than throwing away a dim of size 1. 

# this can be though of as there is actually an additional
# layer after the 10 outputs, except this output only sums
# them together, and the derivative along that path is 1. 

import numpy as np
import pickle
import gzip

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
tr_x = [np.reshape(x, (784, 1)) for x in _tr[0]] 
va_x = [np.reshape(x, (784, 1)) for x in _va[0]]
te_x = [np.reshape(x, (784, 1)) for x in _te[0]]
tr_y = np.array([one_hot_encode(y) for y in _tr[1]])
tr_data = list(zip(tr_x, tr_y))
#va_data = list(zip(va_x, _va[1])) 
te_data = list(zip(te_x, _te[1]))


# params
batch_size = 20
eta = 3.0
np.random.seed(1)
num_inp = 784
num_out = 10
# layer                   0 1 2 3
hidden_structure = [num_inp, 10, 35, 20, num_out]

def mine():
    
    np.random.seed(1)
    B = [None] + [np.random.randn(y, 1) for y in hidden_structure[1:]]
    W = [None] + [np.random.randn(y, x) for x, y in zip(hidden_structure[:-1], hidden_structure[1:])]


    __id3 = np.zeros([batch_size, num_out, num_out])
    __idx = np.arange(num_out)
    __id3[:,__idx,__idx] = 1
    losses = []


    __b = 0
    batch_x = _tr[0][__b*batch_size:(__b + 1)*batch_size,:].T
    batch_y = tr_y[__b*batch_size:(__b + 1)*batch_size,:].T

    # forward pass 
    input = batch_x
    z_s = [None]
    h_s = [batch_x]
    for j in range(1, len(hidden_structure)):
        z = W[j] @ input + B[j]        # weighted input
        input = sigmoid(z)                 # hidden output
        z_s.append(z)
        h_s.append(input)


    # calculate cost
    error = input - batch_y
    losses.append(np.sum(error ** 2))


    jacobians = [np.einsum('ij,il->lij',W[j].T, sigmoidq(h_s[j-1])) for j in range(2, len(hidden_structure))]
    cumul_jacob = [None for i in jacobians] + [__id3] 
    for j in range(-1,-len(jacobians)-1,-1):
        cumul_jacob[j-1] = np.einsum('pij,pjk->pik', jacobians[j], cumul_jacob[j]) # matmul in inner 2 dimensions


    # update weights 
    common = (eta / batch_size) * error * sigmoidq(h_s[-1]) # batch_size scalars
    for j in range(1, len(hidden_structure)):
        B[j] -= np.expand_dims(np.einsum('lj,jkl->k', common, cumul_jacob[j-1]), 1)
        W[j] -= np.einsum('ij,jhi,wj->hw', common, cumul_jacob[j-1], h_s[j-1])

    return B, W


def nielsen():


    np.random.seed(1)
    B = [np.random.randn(y, 1) for y in hidden_structure[1:]]
    W = [np.random.randn(y, x) for x, y in zip(hidden_structure[:-1], hidden_structure[1:])]

    #for _e in range(30): # for each epoch

    #random.shuffle(tr_data) # shuffle data 
    #mini_batches = [tr_data[k:k+batch_size] for k in range(0, len(tr_data), batch_size)] # list of mini_batches
    #for mini_batch in mini_batches: # loop over mini batches

    __b = 0
    batch_x = _tr[0][__b*batch_size:(__b + 1)*batch_size,:].T
    batch_y = tr_y[__b*batch_size:(__b + 1)*batch_size,:].T


    _dB_batch = [np.zeros(b.shape) for b in B] 
    _dW_batch = [np.zeros(w.shape) for w in W] 

    #for x, y in mini_batch: 
    for _p in range(batch_size):

        y = batch_y[:,_p].reshape([10,1])
        x = batch_x[:,_p].reshape([784,1])

        _dB_inst = [np.zeros(b.shape) for b in B] 
        _dW_inst = [np.zeros(w.shape) for w in W] 

        # feedforward
        input = x
        h_s = [x] 
        z_s = [] 
        for b, w in zip(B, W):
            z = w @ input + b
            input = sigmoid(z)
            z_s.append(z)
            h_s.append(input)


        # backward pass
        delta = (h_s[-1] - y.reshape([10,1])) * sigmoidp(z_s[-1]) # grad C wrt y 
        _dB_inst[-1] = delta
        _dW_inst[-1] = delta @ h_s[-2].T 

        for l in range(2, len(hidden_structure)): # cumul_jacob
            z = z_s[-l]
            sp = sigmoidp(z)
            delta = (W[-l+1].T @ delta) * sp
            _dB_inst[-l] = delta
            _dW_inst[-l] = delta @ h_s[-l-1].T
        _dB_batch = [nb + dnb for nb, dnb in zip(_dB_batch, _dB_inst)]
        _dW_batch = [nw + dnw for nw, dnw in zip(_dW_batch, _dW_inst)]


    W = [w - (eta / batch_size) * nw for w, nw in zip(W, _dW_batch)]
    B = [b - (eta / batch_size) * nb for b, nb in zip(B, _dB_batch)]

    return B, W

Bm, Wm = mine()
Bn, Wn = nielsen()
for i in range(len(Bm)-1):
    # compare within some epsilon
    np.all(np.isclose(Bm[i+1], Bn[i]))
    np.all(np.isclose(Wm[i+1], Wn[i]))

