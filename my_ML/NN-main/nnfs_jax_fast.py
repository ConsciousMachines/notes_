# applied jax.lax.scan to the old code and now it's blazing fast.

import pickle, gzip, time
import numpy as onp
import jax 
import jax.numpy as np


# load data
f = gzip.open(r'C:\Users\pwnag\Desktop\sup\nielsen\mnist.pkl.gz', 'rb')
_tr, _va, _te = pickle.load(f, encoding = "latin1")
f.close()
_try = onp.zeros([50000,10])
_try[onp.arange(50000), _tr[1]] = 1 # one hot encode
_trx = np.array(_tr[0]) # train
_try = np.array(_try)
_tex = np.array(_te[0]) # test
_tey = np.array(_te[1])


# params - jax captures globals once
eta = 1.0
batch_size = 10 # needs to be a factor of 50_000 for reshaping
hidden_structure = [784, 30, 10]
onp.random.seed(1)
params = [np.array(onp.random.randn(y)) for y in hidden_structure[1:]] + [np.array(onp.random.randn(y, x)) for x, y in zip(hidden_structure[:-1], hidden_structure[1:])]

###############################################################################
###############################################################################

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def feedforward(params, _x):
    _n = len(hidden_structure) - 1 # jax captures globals once 
    for _i in range(_n):
        _x = sigmoid(np.matmul(params[_i + _n], _x) + params[_i])
    return _x

def loss(params, x, y):
    b1, b2, w1, w2 = params
    reg = np.sum(np.square(w1)) + np.sum(np.square(w2)) + np.sum(np.square(b1)) + np.sum(np.square(b2)) 
    return np.mean(np.square(feedforward(params, x) - y)) + 0.000001 * reg

@jax.jit
def update(params, __idx):
    x = _trx[__idx,:]
    y = _try[__idx,:] 
    g = jax.vmap(lambda x, y: jax.grad(loss)(params, x, y))(x, y)
    return [params[i] - eta * np.sum(g[i], axis = 0) for i in range(len(params))], 0 # return new params 

@jax.jit 
def test_acc(params):
    return np.sum(jax.vmap(lambda x, y: np.equal(np.argmax(feedforward(params, x)), y))(_tex, _tey)) # vmap over test arrays

master_key = jax.random.PRNGKey(42)
keys = jax.random.split(master_key, 100) # replace this part with a permuted set of indices 

for _e in range(30): 

    s = time.time()

    indices = jax.random.permutation(keys[_e], 50_000).reshape(50_000 // batch_size, batch_size) # permute indices 

    params, _ = jax.lax.scan(update, params, indices) # fold update function over indices

    print(f'Epoch\t{_e}\t: {test_acc(params)} / 10000\ttime: {time.time() - s}')

