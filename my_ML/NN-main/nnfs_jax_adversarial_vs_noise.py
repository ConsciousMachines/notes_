# comparing adversarial examples to a network trained with noise added to inputs
# this made the network perform gooder but I lost the magic numbers :(

import pickle, gzip, time
import numpy as onp
import jax 
import jax.numpy as np
import matplotlib.pyplot as plt


# S T E P   1 :   T R A I N   A   B A S I C   M O D E l   T O   U S E   F O R   G E N E R A T I N G   A D V 

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

@jax.grad
def loss(params, x, y): # best: 0.000001
    b1, b2, w1, w2 = params
    reg = np.sum(np.square(w1)) + np.sum(np.square(w2)) + np.sum(np.square(b1)) + np.sum(np.square(b2)) 
    a = feedforward(params, x)
    return np.mean(np.square(a - y)) + 0.000001 * reg 
    
@jax.jit
def update(state, __idx):
    params, k = state
    nk = jax.random.split(k)[1] # next key
    x = _trx[__idx,:]
    #x += jax.random.normal(nk, shape = x.shape) * 0.1 + 0.3
    y = _try[__idx,:]
    g = jax.vmap(lambda _x, _y: loss(params, _x, _y))(x, y)
    return ([params[i] - eta * np.sum(g[i], axis = 0) for i in range(len(params))], nk), nk

@jax.jit 
def test_acc(params):
    return np.sum(jax.vmap(lambda x, y: np.equal(np.argmax(feedforward(params, x)), y))(_tex, _tey)) # vmap over test arrays

mk = jax.random.PRNGKey(42) # master key 
ks = jax.random.split(mk, 30) # one key per epoch to generate random indices  

state = (params, mk)  # our state is neural net params and another key to generate noise

for _e in range(30): 

    s = time.time()
    
    indices = jax.random.permutation(ks[_e], 50_000).reshape(50_000 // batch_size, batch_size)
    
    state, key_seq = jax.lax.scan(update, state, indices) # fold update function over indices
    
    print(f'Epoch\t{_e}\t: {test_acc(state[0])} / 10000\ttime: {time.time() - s}')
    if np.any(np.isnan(state[0][0])): break











# - - - experiments to try: normal nn is almost at 95%
# data augmentation - zoom +-20%, width, height, rotate +-10% or +- 36 degrees, translate, shear
# normalize mnist to -1,1 -> accomplished nothing. gave nans. doing just the mean changed nothing. 
# train with noise 
#     - i can make the noise super tiny by making lambda super big (1000.0), and it usually 
#       converges after 1 training loop. lol. the noise is usually e-04 or e-05 now. 
#       maybe training on this noise, but also larger may be beneficial. 
#     - with a good (0.000001) regularization rate it seems to work, no nans, and generalizes to the test set. 
#       IT WORkS!!! although a few magic numbers are involved. 
#       first get a normal NN to train. use it to generate adv examples. I used a lambda of 1000.0 
#       while the original guy used 0.5 lol. then retrain the network with the "add noise line". 
#       then doing feedforward will show the original was like 1_515 but the noise-NN got 45_000 !!!


#          A D V E R S A R I A l   E X A M P l E 

j_tr1 = np.array(_tr[1]) # the y labels in jax mode
params, _k = state


def one_hot_encode(i):
    e = np.zeros(10)
    return e.at[i].set(1)


def show_adv(x, y): # https://www.pythonpool.com/matplotlib-cmap/
    figure, axis = plt.subplots(1,2)
    axis[0].imshow(x.reshape(28, 28), cmap='plasma')
    axis[1].bar([0,1,2,3,4,5,6,7,8,9], y)
    figure.set_figwidth(8)
    figure.set_figheight(3)
    plt.show()


def generate_adv(i): # magic numbers: lambda = 1.0, eta = 10.0, run for 200 steps
    # i sat and cycled thru these hyper-params one by one, increasing and decreasing it, keeping the one w lowest recognition rate
    # so these hyper params are overfit to my specific model above, which is itself overfit to the random seed 
    adv_prior = _trx[i]                        # the real number
    real_answer = j_tr1[i]                    # what its real value is 
    fake_answer = (real_answer + 7) % 10
    y = one_hot_encode(fake_answer) # choose it to be anything but the real_answer

    def loss_adv(x): # params fixed in this scenario 
        return np.mean(np.square(feedforward(params, x) - y)) + 1.0 * np.mean(np.square(x - adv_prior))

    def update_adv(x, i):
        return x - 10.0 * jax.grad(loss_adv)(x), 0

    x, _ = jax.lax.scan(update_adv, np.zeros([784]), np.arange(100))

    #show_adv(x, y)
    return x # the corresponding answers are just _try

vmap_generate_adv = jax.jit(jax.vmap(generate_adv))
advs = vmap_generate_adv(np.arange(50000))

i = 30
show_adv(advs[i], feedforward(params, advs[i]))

params, _k = state

def compare_real_vs_fake(i):
    return j_tr1[i] == np.argmax(feedforward(params, advs[i]))

compare_all = jax.jit(jax.vmap(compare_real_vs_fake))
np.sum(compare_all(np.arange(50000)))





# S T E P   3 :   T R A I N   M O D E l   W I T H   N O I S E 
# the only line that's different here from the NN above is the one adding random noise 




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

def loss(params, x, y): # best: 0.000001
    b1, b2, w1, w2 = params
    reg = np.sum(np.square(w1)) + np.sum(np.square(w2)) + np.sum(np.square(b1)) + np.sum(np.square(b2)) 
    a = feedforward(params, x)
    return np.mean(np.square(a - y)) + 0.000001 * reg 
    
@jax.jit
def update(state, __idx):
    params, k = state
    nk = jax.random.split(k)[1] # next key
    x = _trx[__idx,:]
    x += jax.random.normal(nk, shape = x.shape) * 0.1 + 0.3
    y = _try[__idx,:]
    g = jax.vmap(lambda _x, _y: jax.grad(loss)(params, _x, _y))(x, y)
    return ([params[i] - eta * np.sum(g[i], axis = 0) for i in range(len(params))], nk), nk

@jax.jit 
def test_acc(params):
    return np.sum(jax.vmap(lambda x, y: np.equal(np.argmax(feedforward(params, x)), y))(_tex, _tey)) # vmap over test arrays

mk = jax.random.PRNGKey(42) # master key 
ks = jax.random.split(mk, 30) # one key per epoch to generate random indices  

state = (params, mk)  # our state is neural net params and another key to generate noise

for _e in range(30): 

    s = time.time()
    
    indices = jax.random.permutation(ks[_e], 50_000).reshape(50_000 // batch_size, batch_size)
    
    state, key_seq = jax.lax.scan(update, state, indices) # fold update function over indices
    
    print(f'Epoch\t{_e}\t: {test_acc(state[0])} / 10000\ttime: {time.time() - s}')
    if np.any(np.isnan(state[0][0])): break



# S T E P   4 :   R U N   N O I S E   M O D E l   O N   A D V 
params, _k = state

def compare_real_vs_fake(i):
    return j_tr1[i] == np.argmax(feedforward(params, advs[i]))

compare_all = jax.jit(jax.vmap(compare_real_vs_fake))
np.sum(compare_all(np.arange(50000)))



# step 5: what if we generate a network on adversarial examples (and noise), and then try make 
# adversarial examples from it?