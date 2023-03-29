# old inferior version!!! i found a better way later when considering a vector output
# EINSTEIN NOTATION FOR CHADS. does this solve the matrix problem? not really, but I feel empowered.
# NOTE: there is an exposition below on how it works. 

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
eta = -2.
batch_size = 4


# init weights 
np.random.seed(0)
# layer                   0 1 2 3
hidden_structure = [num_inp,17,5,1]
W = [None] # so that weights of layer i will be at index i
B = [None]
for i in range(1, len(hidden_structure)):
    W.append(np.random.uniform(size=[hidden_structure[i], hidden_structure[i-1]]))
    B.append(np.random.uniform(size=[hidden_structure[i], 1]))


losses = []
for ii in range(2500):


    # forward pass 
    input = X.T
    z_s = [None]
    h_s = [X.T]
    for j in range(1, len(hidden_structure)):
        z = W[j] @ input + B[j]        # weighted input
        h = sigmoid(z)                 # hidden output
        input = h                      # propagate this to next layer's input 
        z_s.append(z)
        h_s.append(h)


    # calculate cost
    error = input - Y.T
    losses.append(np.sum(error ** 2))


    # takes each column of z_s[j-1], multiplies it broadcasting element wise by W[j].T, and stacks results. work it out on paper
    jacobians = [np.einsum('ij,il->lij',W[j].T, sigmoidq(h_s[j-1])) for j in range(2, len(hidden_structure))]
    cumul_jacob = [None for i in jacobians] + [np.ones([batch_size,1,1])] # the last is jacobian grad_H4_wrt_H4
    for j in range(-1,-len(jacobians)-1,-1):
        cumul_jacob[j-1] = np.einsum('pij,pjk->pik', jacobians[j], cumul_jacob[j]) # matmul in inner 2 dimensions


    # update weights 
    common = eta * error * sigmoidq(h_s[-1]) # batch_size scalars
    for j in range(1, len(hidden_structure)):
        B[j] += np.einsum('ij,jkl->kl', common, cumul_jacob[j-1])
        W[j] += np.einsum('qb,bij,ob->io', common, cumul_jacob[j-1], h_s[j-1])




plt.plot(losses)
plt.ylim(0, 2)
plt.show()




''' 
# E X P O S I T I O N   O N   E I N S T E I N   S U M M A T I O N
# before i had 
common = [[1,2,3,4]] # 1x4 
yep = [[[1],[2],[3]],[[1],[2],[3]],[[1],[2],[3]],[[1],[2],[3]]] # 4x3x1 
'ij,jkl->kl' # unrolled to:
# i = 1 thrown away (or added over? doesnt matter since it's 1)
# j = 4 looped and multiplied and added
# k = 3 kept as result axis 0
# l = 1 kept as result axis 1 
res = np.zeros([3,1])
for k in range(3): # loop over result's axis 0
    for l in range(1): # loop over result's axis 1
        accum = 0 # this is where result of j will go 
        for j in range(4):
            accum += common[0][j] * yep[j][k][l]
        res[k][l] = accum
    


# now i have
common # 10x4 :: 10 outputs by 4 batch
yep = cumul_jacob[j-1] # 4x30x10 :: 4 batch by 30 grads by 10 outputs 
# what do i want to do? sum everything into one 30x1 array.. or just 30 for now since i cant find a 1 anywhere
_res = np.zeros(30)
for _i in range(30):
    # each element should contain the 4x10 thing at yep multiplied by the 4x10 thing in common, and summed all over 
    #yep[:,_i,:] # this is the 4x10 thing in yep
    #common # is already 10x4
    # now i need to iterate over 10 and over 4 and sum all 
    accum1 = 0
    for _j in range(10):
        for _k in range(4):
            accum1 += common[_j,_k] * yep[_k,_i,_j]
    _res[_i] = accum1

_res == np.einsum('lj,jkl->k', common, cumul_jacob[j-1]) # it works up to numerical imprecision!





# then i had 
j = 1
W[j].shape

common.shape           # 10x4
soy = cumul_jacob[j-1] # 4x30x10
boy = h_s[j-1]         # 784x4
# output needs to be 30x784 
_res = np.zeros([30,784])
for _i in range(30):
    for _j in range(784):
        # firstly we are doing a matrix multiply over a 30x1 matrix and a 1x784 matrix
        # for each entry we want to accumulate stuff over the 10, 4 dims
        accum = 0
        for _l in range(4): # we accumulate over the 4 axis on all of them 
            for _k in range(10): # but accumulate over the 10 axis on only two 
                accum += common[_k,_l] * soy[_l,_i,_k] * boy[_j,_l]
        _res[_i,_j] = accum

_res == np.einsum('ij,jhi,wj->hw', common, cumul_jacob[j-1], h_s[j-1])
'''

