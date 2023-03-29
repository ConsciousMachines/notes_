# a first attempt at adversarial, which requires lots of hyper parameter tuning. 

import pickle, gzip, time
import numpy as onp
import jax 
import jax.numpy as np
import matplotlib.pyplot as plt


# data augmentation - zoom +-20%, width, height, rotate +-10% or +- 36 degrees, translate, shear


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
eta = 0.5
batch_size = 20 # needs to be a factor of 50_000 for reshaping
hidden_structure = [784, 30, 10]
onp.random.seed(1)
params = [onp.random.randn(y) for y in hidden_structure[1:]] + [onp.random.randn(y, x) for x, y in zip(hidden_structure[:-1], hidden_structure[1:])]

master_key = jax.random.PRNGKey(42)
keys = jax.random.split(master_key, 100) # replace this part with a permuted set of indices 

###############################################################################
###############################################################################


def feedforward(params, _x):
    _n = len(hidden_structure) - 1 # jax captures globals once 
    for _i in range(_n):
        _x = 1.0 / (1.0 + np.exp(-(np.matmul(params[_i + _n], _x) + params[_i]))) # this is a bit faster than having separate sigmoid func
    return _x

def loss(params, x, y):
    return np.mean(np.square(feedforward(params, x) - y)) # sum gives nans :(

@jax.jit
def update(params, __idx):
    x = _trx[__idx,:]                                                                # batch_size x 784
    # TODO: add noise to x (requires key)
    y = _try[__idx,:]                                                                # batch_size x 10
    g = jax.vmap(lambda x, y: jax.grad(loss)(params, x, y))(x, y)                    # vmap grad calculation over batch dimension
    return [params[i] - eta * np.sum(g[i], axis = 0) for i in range(len(params))], 0 # return new params 

@jax.jit 
def test_acc(params):
    return np.sum(jax.vmap(lambda x, y: np.equal(np.argmax(feedforward(params, x)), y))(_tex, _tey)) # vmap over test arrays

for _e in range(30): 

    s = time.time()

    indices = jax.random.permutation(keys[_e], 50_000).reshape(50_000 // batch_size, batch_size) # permute indices 

    params, _ = jax.lax.scan(update, params, indices) # fold update function over indices

    print(f'Epoch\t{_e}\t: {test_acc(params)} / 10000\ttime: {time.time() - s}')





#          A D V E R S A R I A l   E X A M P l E 


def one_hot_encode(i):
    e = onp.zeros(10)
    e[i] = 1
    return np.array(e)


def show_adv(x, y): # https://www.pythonpool.com/matplotlib-cmap/
    figure, axis = plt.subplots(1,2)
    axis[0].imshow(x.reshape(28, 28), cmap='plasma')
    axis[1].bar([0,1,2,3,4,5,6,7,8,9], y)
    figure.set_figwidth(8)
    figure.set_figheight(3)
    plt.show()


def generate_adv(i):

    adv_prior = _trx[i]                        # the real number
    real_answer = _tr[1][i]                    # what its real value is 
    y = one_hot_encode((real_answer + 7) % 10) # choose it to be anything but the real_answer

    def loss_adv(x): # params fixed in this scenario 
        return np.mean(np.square(feedforward(params, x) - y)) + np.mean(np.square(x - adv_prior))

    @jax.jit
    def update_adv(x, i):
        return x - eta * jax.grad(loss_adv)(x), 0

    x, _ = jax.lax.scan(update_adv, np.zeros([784]), np.arange(500))

    show_adv(x, y)


generate_adv(32)
