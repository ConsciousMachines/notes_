# applying my network to make adversarial examples with MNIST. 

import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt

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
batch_size = 16
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

    for _p in range(50000 // batch_size):

        # create batch
        __idx = np.random.choice(np.arange(50000),batch_size)
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
    print("Epoch {} : {} / {}".format(_e, __evaluate, len(te_data)))



_net_spec = 'i_30_o'
with open(r'C:\Users\pwnag\Desktop\sup\deep_larn\mnist_W_' + _net_spec, 'wb') as _f:
    pickle.dump(W,_f)
with open(r'C:\Users\pwnag\Desktop\sup\deep_larn\mnist_B_' + _net_spec, 'wb') as _f:
    pickle.dump(B,_f)






#          A D V E R S A R I A l   E X A M P l E 
__id3adv = np.zeros([1, num_out, num_out]) # we need to update this thing because it relies on batch_size
__id3adv[:,__idx,__idx] = 1

def show_digit(digit_array): # https://www.pythonpool.com/matplotlib-cmap/
    plt.imshow(digit_array.reshape(28, 28), cmap='plasma')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def belief_graph(input):
    plt.bar([0,1,2,3,4,5,6,7,8,9], input)
    plt.show()


lam = 0.05
adv_want = one_hot_encode(1).reshape([10,1])  # desired output: 1
adv_prior = _tr[0][17,:].reshape([784,1])     # an 8
x = np.zeros([784,1]) # start 

for _p in range(1000):

    # forward pass 
    input = x
    z_s = [None]
    h_s = [x]
    for j in range(1, len(hidden_structure)):
        z = W[j] @ input + B[j]        
        input = sigmoid(z)                
        z_s.append(z)
        h_s.append(input)

    # get the jacobians and gradients
    jacobians = [np.einsum('ij,il->lij',W[j].T, sigmoidq(h_s[j-1])) for j in range(2, len(hidden_structure))]
    cumul_jacob = [None for i in jacobians] + [__id3adv] 
    for j in range(-1,-len(jacobians)-1,-1):
        cumul_jacob[j-1] = np.einsum('pij,pjk->pik', jacobians[j], cumul_jacob[j]) # matmul in inner 2 dimensions

    # update x
    error = input - adv_want
    grad_C_wrt_X = np.expand_dims(np.einsum('aq,io,qia->o', error, W[1], cumul_jacob[0]), 1) # W[1].T @ cumul_jacob[0].squeeze() @ error
    x -= grad_C_wrt_X + lam * (x - adv_prior) # regularize to be adversarial image 




#show_digit(adv_prior)
#belief_graph(adv_want.squeeze())
show_digit(x)
belief_graph(input.squeeze())
