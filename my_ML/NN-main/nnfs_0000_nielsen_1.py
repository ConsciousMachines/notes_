# I took apart Nielsen's NN code. it turned out to be very similar to what I made
# I thought there was some crazy matrix algebra for batching. no. its just a cheap for loop

# then I made sure my output matches his for a single batch. 
# they did except I had to divide dW, dB by 1/batch_size

# then I made a better batching system that uses numpy indices (which r fast)
# now this code is a fusion of his starter code and some of my modifications
# his backward pass is still his and mine is still einsum ballerswaggin_v2

# interestingly, after all that, they get slightly different evaluation results.
# I saw that even after a single pass their answers are different enough that 
# I had to use np.close to calculate them. I guess the numerical imprecision
# depending on the math formula really adds up quick. 

import pickle
import gzip
import numpy as np

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
    for b, w in zip(B, W):
        a = sigmoid(w @ a + b)
    return a

###############################################################################
###############################################################################




# params
batch_size = 9
eta = 3.0
np.random.seed(1)
num_inp = 784
num_out = 10
# layer                   0 1 2 3
hidden_structure = [num_inp,30,num_out]
B = [np.random.randn(y, 1) for y in hidden_structure[1:]]
W = [np.random.randn(y, x) for x, y in zip(hidden_structure[:-1], hidden_structure[1:])]


for _e in range(30): # for each epoch

    for _p in range(50000 // batch_size): # how many batches do we need

        _dB_batch = [np.zeros(b.shape) for b in B] # grad B for batch
        _dW_batch = [np.zeros(w.shape) for w in W] # grad W for batch

        # create my batch
        __idx = np.random.choice(np.arange(50000),batch_size)
        batch_x = _tr[0][__idx,:].T
        batch_y = tr_y[__idx,:].T

        for _i in range(batch_size): # my version with my batch
            x = batch_x[:,_i].reshape([784,1])
            y = batch_y[:,_i]
            
            _dB_inst = [np.zeros(b.shape) for b in B] # grad B for inst
            _dW_inst = [np.zeros(w.shape) for w in W] # grad W for inst

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
        
    # test on te_data
    __test_results = [(np.argmax(feedforward(x)), y) for (x, y) in te_data]
    __evaluate = sum(int(x == y) for (x, y) in __test_results)
    print("Epoch {} : {} / {}".format(_e, __evaluate, len(te_data)))

