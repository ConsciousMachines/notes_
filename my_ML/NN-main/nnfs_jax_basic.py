# slow version, simply added jax grad to my original MNIST NN. 
# also used vmap to parallelize testing and batch.

import pickle, gzip, time
import numpy as onp
import jax 
import jax.numpy as np

# load data
f = gzip.open(r'C:\Users\pwnag\Desktop\sup\nielsen\mnist.pkl.gz', 'rb')
_tr, _va, _te = pickle.load(f, encoding = "latin1")
f.close()
tr_y = onp.zeros([50000,10])
tr_y[onp.arange(50000), _tr[1]] = 1 # one hot encode
tr_x = np.array(_tr[0]) # train
tr_y = np.array(tr_y)
_tex = np.array(_te[0]) # test
_tey = np.array(_te[1])

# params - jax captures globals once
eta = 1.0
batch_size = 10
hidden_structure = [784, 30, 10]
onp.random.seed(1)
params = [onp.random.randn(y) for y in hidden_structure[1:]] + [onp.random.randn(y, x) for x, y in zip(hidden_structure[:-1], hidden_structure[1:])]

###############################################################################
###############################################################################



def feedforward(params, _x):
    _n = len(hidden_structure)-1 # jax captures globals once 
    for _i in range(_n):
        _x = 1.0 / (1.0 + np.exp(-(np.matmul(params[_i + _n], _x) + params[_i]))) 
    return _x

jit_feedforward = jax.jit(feedforward)

def loss(params, x, y):
    return np.mean(np.square(feedforward(params, x) - y)) 

grad_loss = jax.jit(jax.grad(loss))


for _e in range(30): 

    s = time.time()
    
    for _p in range(50000 // batch_size):

        # create batch
        __idx = onp.random.choice(50000, batch_size)
        x = tr_x[__idx,:]
        y = tr_y[__idx,:]

        # get grads
        grads = jax.vmap(lambda x, y: grad_loss(params, x, y))(x, y)

        # update
        for i in range(len(params)):
            params[i] = params[i] - eta * np.sum(grads[i], axis=0)


    # test accuracy
    acc = np.sum(jax.vmap(lambda x, y: np.equal(np.argmax(jit_feedforward(params, x)), y))(_tex, _tey))
    print(f'Epoch\t{_e}\t: {acc} / 10000\ttime: {time.time() - s}')







