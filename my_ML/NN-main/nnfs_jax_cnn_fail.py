# Vsevolod Ladtchenko 20895137
# stat 940 data challenge 1


# ------------------------------------------------
# Part 1 - load the data
# ------------------------------------------------

import numpy as onp
from PIL import Image as im
import os, pickle

# save model
def save_it(thing, path): 
    with open(path, 'wb') as f:
        pickle.dump(thing, f)

# load model
def load_it(path): 
    with open(path, 'rb') as f:
        return pickle.load(f)

# zipped version 
def load_data(path):
    import zipfile, io
    zip = zipfile.ZipFile(path)
        
    # get the test vs train files 
    files = zip.namelist()
    trains = []
    tests = []
    for i in range(len(files)):
        one_file = files[i]
        if 'test/test/' in one_file:
            tests.append(one_file)
        elif 'train/train/' in one_file:
            trains.append(one_file)
        else:
            print(one_file)

    # the two non-image files 
    with zip.open('sample_submission.csv') as f:
        sample_submission = f.read()
    with zip.open('train_labels.csv') as f:
        f = zip.open('train_labels.csv')
        labels = onp.zeros(50_000, dtype = onp.int32)
        for i, line in enumerate(f.readlines()[1:]): # skip first row 
            labels[i] = int(str(line).split(',')[-1][:1])
    print(f'labels mean: {labels.mean()}')

    Y = onp.zeros([50_000,10])
    Y[onp.arange(50_000), labels] = 1 # one hot encode

    X = onp.zeros([50_000,32,32,3], dtype = onp.uint8)
    for i in range(len(trains)):
        real_index = int(trains[i].split('/')[2].split('.')[0])
        X[real_index,:,:,:] = onp.array(im.open(io.BytesIO(zip.open(trains[i]).read())), dtype = onp.uint8)

    X_test = onp.zeros([10_000,32,32,3], dtype = onp.uint8)
    for i in range(len(tests)):
        real_index = int(tests[i].split('/')[2].split('.')[0])
        X_test[real_index,:,:,:] = onp.array(im.open(io.BytesIO(zip.open(tests[i]).read())), dtype = onp.uint8)

    return X, Y, labels, X_test, sample_submission



# load data
def load_data():
        
    train_dir                              = r'C:\Users\pwnag\Desktop\sup\deep_larn\train\train'
    test_dir                               = r'C:\Users\pwnag\Desktop\sup\deep_larn\test\test'
    labels_dir                             = r'C:\Users\pwnag\Desktop\sup\deep_larn\train_labels.csv'

    train_dir                              = r'../input/940-data-1/train/train'
    test_dir                               = r'../input/940-data-1/test/test'
    labels_dir                             = r'../input/940-data-1/train_labels.csv'

    __labels                               = onp.loadtxt(labels_dir, skiprows=1,dtype=onp.int32,delimiter=',')[:,1]
    __Y                                    = onp.zeros([50_000,10])
    __Y[onp.arange(50_000), __labels]      = 1 # one hot encode

    def load_pics(dir):
        files                              = os.listdir(dir)
        ___X                               = onp.zeros([len(files),32,32,3], dtype = onp.uint8)
        for i in range(len(files)):
            real_index                     = int(files[i].split('/')[-1].split('.')[0])
            ___X[real_index,:,:,:]         = onp.array(im.open(os.path.join(dir, files[i])), dtype = onp.uint8)
        return ___X
    __X                                    = load_pics(train_dir)
    __X_kaggle                             = load_pics(test_dir)
    return __X, __Y, __labels, __X_kaggle

# this is a function to augment X by rotating the images by -20,-10,10,20 degrees. The 
# resulting data set was too large and caused jax to crash so I didn't end up using it.
# the alternative was to make a system that rotates inputs at runtime, but I didn't have time.
def augment_data(_X, _Y, _labels): 
    _X2                                    = onp.zeros([200_000, 32, 32, 3], dtype=onp.uint8)
    for i in range(50_000):
        img                                = im.fromarray(_X[i,:,:,:])
        _X2[i          ,:,:,:]             = onp.array(img.rotate(-20, im.BILINEAR, expand = 0),dtype=onp.uint8)
        _X2[i + 50_000 ,:,:,:]             = onp.array(img.rotate(-10, im.BILINEAR, expand = 0),dtype=onp.uint8)
        _X2[i + 100_000,:,:,:]             = onp.array(img.rotate( 10, im.BILINEAR, expand = 0),dtype=onp.uint8)
        _X2[i + 150_000,:,:,:]             = onp.array(img.rotate( 20, im.BILINEAR, expand = 0),dtype=onp.uint8)
    _X                                     = onp.concatenate([_X,_X2],0)
    _Y                                     = onp.concatenate([_Y for i in range(5)], 0)
    _labels                                = onp.concatenate([_labels for i in range(5)])
    return _X, _Y, _labels

# these are the 'raw' or original data, not for processing
_X, _Y, _labels, _X_kaggle                 = load_data()
#_X, _Y, _labels = augment_data(_X, _Y, _labels)


# check that our data set works
import matplotlib.pyplot as plt
plt.imshow(_X[-1,:,:,:])
#plt.show()







# ------------------------------------------------
# Part 2 - prepare the data
# ------------------------------------------------

# the base code I used came from here:
# https://coderzcolumn.com/tutorials/artifical-intelligence/jax-guide-to-create-convolutional-neural-networks#2

import jax
from jax.example_libraries import stax, optimizers
from jax import numpy as np

# this is the entire data, ready for processing 
full_X      = np.array(_X, dtype=np.float32) / 255.0          
full_Y      = np.array(_Y, dtype=np.float32)
full_labels = np.array(_labels)

# split the data into train & test sets
n           = full_X.shape[0]                                                        # size of data
n9          = int(0.9 * n)                                                           # size of 90% of data
onp.random.seed(0)                                                                   # set seed before random ops
__idx       = onp.random.choice(n, n9, replace = False)                              # randomly pick 90% of indices
__idxt      = onp.setdiff1d(onp.arange(n), __idx)                                    # the rest for test
X           = full_X[__idx,:,:,:]                                                    # the train set
Y           = full_Y[__idx]
labels      = full_labels[__idx]
Xt          = full_X[__idxt,:,:,:]                                                   # the test set
Yt          = full_Y[__idxt]
labelst     = full_labels[__idxt]

del full_X, full_Y, full_labels, _X, _Y, _labels, _X_kaggle








# ------------------------------------------------
# Part 3 - Network design 
# ------------------------------------------------

# now that the data is split into train and test sets, I can look for an architecture.
# the plan is to find one that works well on that random test set and then train it 
# on the entire set before applying to the kaggle competition.

# to avoid overfitting I first experimented with general heuristics, described below,
# rather than tweak learning parameters which guarantees nothing.  

# The goal was to start with a simple Conv(32),Relu,Conv(16),Relu network and
# make small changes to observe the output. The hypothesis was that the first
# convolutional layer, with a filter size of 3 and stride of 1, will see 3 pixels
# in either direction. Then the second layer will see 9, third will see 27, and 
# fourth will see 81. So I decided the model should have at least 4 layers in order
# for the last layer to be able to potentially capture signals from opposite corners
# of the image, which is 32x32. Trying this out empirically has shown that 3 layers 
# always perform worse and 5 layers do not perform better than 4. 

# I added batch norm early on as it is generally helpful, results agreed. Adding 
# MaxPool has sped up training as it divides each image dimension by 2, and reduces
# correlation by picking the most interesting signal. Relu was used because the input 
# is normalized to [0,1], and we do not want to cancel out some features with others
# by having a negative output. Before adding dropout, I noticed the train accuracy 
# quickly rose to 99 while test was some distance under it. With dropout this 
# distance grew slower, but it did not help increase test accuracy. I uncluded
# L2 regularization to prevent NaNs. the last layer is log of softmax, because
# we use a cross entropy loss since this is a classification problem. 

# I added Gaussian noise * 0.05 to the images at runtime (different each batch)
# so about 12.5 std in pixel brightness. In a previous experiment I found that 
# adding noise while training a network to recognize MNIST handwritten digits 
# will greatly help prevent adversarial attacks. So maybe this helps the network
# learn higher level features instead of specific pixel combinations at low layers.

# at this point the only thing left to do is make the network wider. Each time I 
# increase the width I get better results. Using less neurons in later layers decreased
# test accuracy, which is an intriguing observation. 

# this concludes the heuristic design of the network. What is left is to change the 
# optimization algorithm (SGD vs Adam), its learning rate, batch size, and weight
# regularizer. This is all random, unexplainable, and overfits to my test set.
# one potential extension would be to train a few models with different hyper 
# parameters to create an ensemble, but each model now take about 50 minutes. 
# at this point I am getting 90% test accuracy most epochs. 

conv_init, conv_apply = stax.serial(
    stax.Conv(1024, (3,3), (1,1)   padding = 'SAME'),
    stax.BatchNorm(),
    stax.Relu,
    stax.MaxPool((2,2), (2,2),     padding = 'SAME'),
    stax.Dropout(0.8,              mode    = 'train'),
    stax.Conv(1024, (3,3), (1,1)   padding = 'SAME'),
    stax.BatchNorm(),
    stax.Relu,
    stax.MaxPool((2,2), (2,2),     padding = 'SAME'),
    stax.Dropout(0.8,              mode    = 'train'),
    stax.Conv(1024, (3,3), (1,1)   padding = 'SAME'),
    stax.BatchNorm(),
    stax.Relu,
    stax.MaxPool((2,2), (2,2),     padding = 'SAME'),
    stax.Dropout(0.8,              mode    = 'train'),
    stax.Conv(1024, (3,3), (1,1)   padding = 'SAME'),
    stax.BatchNorm(),
    stax.Relu,
    stax.MaxPool((2,2), (2,2),     padding = 'SAME'),
    stax.Dropout(0.8,              mode    = 'train'),

    stax.Flatten,
    stax.Dense(10),
    stax.LogSoftmax
)

# learning / hyper parameters 
batch_size = 256
epochs     = 100
alpha      = 0.0001                                                                  # weight regularizer
eta        = 0.001                                                                   # learning rate of SGD or Adam
OPT        = 'sgd' # 'adam' or 'sgd'


@jax.jit
def update(state, __idx):
    # Globals: X, Y, CrossEntropyLoss, opt_get_w, opt_update
    opt_state, k, step = state                                                       # unpack passed state into optimizer-state, key, step
    nk                 = jax.random.split(k)[1]                                      # get next key from old key
    x, y               = X[__idx], Y[__idx]                                          # get the batch using the indices
    x                  = x + jax.random.normal(nk, shape = x.shape) * 0.05           # add noise 
    x                  = np.maximum(0, x)                                            # make sure the image isn't below 0
    g                  = jax.grad(CrossEntropyLoss)(opt_get_w(opt_state), x, y, nk)  # gradient step
    opt_state          = opt_update(step, g, opt_state)                              # update parameters using optimizer
    return (opt_state, nk, step + 1), None                                           # returns next state (None is for jax.lax.scan)


def CrossEntropyLoss(wts, x, y, k):                                                  # the conv net conv_apply requires random key, k, for Dropout 
    # Globals: conv_apply, alpha
    reg = 0                                                                          # regularization cost counter
    for w in wts:                                                                    # wts is a list of tuples, some empty, containing weights and biases
        if w:                                                                        # check if empty
            reg += np.sum(w[0] * w[0]) + np.sum(w[1] * w[1])                         # add squared sum of all weights and biases
    return - np.sum(y * conv_apply(wts, x, rng=k)) + alpha * reg                     # return cross entropy plus L2 regularization cost 


@jax.jit
def generate_random_indices(_e):                                                     # generate a permutation of indices for any batch size 
    # Globals: X, batch_size, ks
    _n          = X.shape[0]                                                         # training data size 
    num_batches = (_n // batch_size) + 1                                             # number of batches needed to cover all data
    perm        = np.expand_dims(jax.random.permutation(ks[_e], _n), axis = 0)       # random permutation of the training data indices 
    more        = jax.random.choice(ks[_e], _n, [1, num_batches * batch_size - _n])  # pick more indices to fill up num_batches x batch_size
    return np.concatenate([perm, more], 1).reshape([num_batches, batch_size])        # return matrix num_batches x batch_size of random indices


def accuracy(wts, __X, __labels):                                                    # jitting this causes bug in jax 
    # Globals: batch_size, conv_apply, mk
    c          = 0                                                                   # counter
    n          = __X.shape[0]                                                        # size of given data set 
    for i in range(n // batch_size):                                                 # loop over all batch_size batches
        _start = i * batch_size                                                      # start index for that batch
        _end   = (i + 1) * batch_size                                                # end index for that batch
        pred   = conv_apply(wts, __X[_start:_end,:,:,:], rng=mk)                     # predictions for that batch 
        c      = c + np.sum(np.argmax(pred, 1) == __labels[_start:_end])             # do the predictions match the labels?
    if _end   != n:                                                                  # now do the chunk of data left over 
        pred   = conv_apply(wts, __X[_end:,:,:,:], rng=mk)                           # this chunk is usually less than batch_size
        c      = c + np.sum(np.argmax(pred, 1) == __labels[_end:])
    return c                                                                         # return the number predicted correct 



mk                                    = jax.random.PRNGKey(42)                       # master key, used for RNG and Dropout
ks                                    = jax.random.split(mk, epochs)                 # one key per epoch to generate random indices  
wts                                   = conv_init(mk, (batch_size,32,32,3))[1]       # net parameters: list of tuples of w,b, some empty
if OPT == 'adam':
    opt_init, opt_update, opt_get_w   = optimizers.adam(eta) 
elif OPT == 'sgd':
    opt_init, opt_update, opt_get_w   = optimizers.sgd(eta)
opt_state                             = opt_init(wts)                                # initialize optimizer state with our net parameters
state                                 = (opt_state, mk, 0)                           # pack state with key (for Dropout), and step (for optimizer)

for _e in range(1, epochs+1):                                                        # for each epoch

    indices              = generate_random_indices(_e)                               # generate random indices

    state, _             = jax.lax.scan(update, state, indices)                      # fold update function over indices (same as for loop)

    opt_state, __, _     = state                                                     # unpack the state

    wts                  = opt_get_w(opt_state)                                      # get the weights from the state


    acc                  = accuracy(wts, Xt, labelst)/Xt.shape[0]                    # test accuracy
    if acc > 0.90:                                                                   # if accuracy > 90%, save weights
        _precision       = '%.4f' % acc
        file_name        = f'wts_{_precision}_{batch_size}_{_e}_{alpha}_{eta}_{OPT}'
        full_path        = os.path.join(os.getcwd(), file_name)
        save_it(wts, full_path)
    print(f'Epoch: {_e}\tTest Acc: {acc}')                                           # print test accuracy for this epoch

print('done')








# ------------------------------------------------
# Part 4 - train on full set 
# ------------------------------------------------

# option 1: retrain the whole network
if False:
        
    # to train the network on the entire training set, we have to make X, Y, labels 
    # be the entire data set. Then we have to re-run Part 3 because the jax-jitted 
    # functions only capture the values of global variables the first time they are 
    # called, and they remain unchanged until the function is re-compiled.

    _X, _Y, _labels, _X_kaggle = load_data()

    full_X                     = np.array(_X, dtype=np.float32) / 255.0          
    full_Y                     = np.array(_Y, dtype=np.float32)
    full_labels                = np.array(_labels)

    X                          = full_X[:,:,:,:]                          
    Y                          = full_Y[:]
    labels                     = full_labels[:]

    del full_X, full_Y, full_labels, _X, _Y, _labels, _X_kaggle


# option 2: train the ready network on what we previously used for testing
# first we set X, Y, and labels to be the unseen data. Then we have to re-run
# a couple of functions which capture the global X, so they capture the new one.
# then we run a training loop of 100 epochs for the smaller data set. 

X      = Xt
Y      = Yt
labels = labelst


@jax.jit
def update(state, __idx):
    # Globals: X, Y, CrossEntropyLoss, opt_get_w, opt_update
    opt_state, k, step = state                                                       # unpack passed state into optimizer-state, key, step
    nk                 = jax.random.split(k)[1]                                      # get next key from old key
    x, y               = X[__idx], Y[__idx]                                          # get the batch using the indices
    x                  = x + jax.random.normal(nk, shape = x.shape) * 0.05           # add noise 
    x                  = np.maximum(0, x)                                            # make sure the image isn't below 0
    g                  = jax.grad(CrossEntropyLoss)(opt_get_w(opt_state), x, y, nk)  # gradient step
    opt_state          = opt_update(step, g, opt_state)                              # update parameters using optimizer
    return (opt_state, nk, step + 1), None                                           # returns next state (None is for jax.lax.scan)


@jax.jit
def generate_random_indices(_e):                                                     # generate a permutation of indices for any batch size 
    # Globals: X, batch_size, ks
    _n          = X.shape[0]                                                         # training data size 
    num_batches = (_n // batch_size) + 1                                             # number of batches needed to cover all data
    perm        = np.expand_dims(jax.random.permutation(ks[_e], _n), axis = 0)       # random permutation of the training data indices 
    more        = jax.random.choice(ks[_e], _n, [1, num_batches * batch_size - _n])  # pick more indices to fill up num_batches x batch_size
    return np.concatenate([perm, more], 1).reshape([num_batches, batch_size])        # return matrix num_batches x batch_size of random indices


for _e in range(1, epochs + 1):                                                      # for each epoch
    indices              = generate_random_indices(_e)                               # generate random indices
    state, _             = jax.lax.scan(update, state, indices)                      # fold update function over indices (same as for loop)
    opt_state, __, _     = state                                                     # unpack the state
    wts                  = opt_get_w(opt_state)                                      # get the weights from the state
    print(_e)
print('done')




# ------------------------------------------------
# Part 5 - write kaggle prediction
# ------------------------------------------------

def do_kaggle(write_dir, _X_kaggle, batch_size, conv_apply, wts, mk):
    # Globals: 0
    X_kaggle    = np.array(_X_kaggle, dtype=np.float32) / 255.0                      # kaggle input data
    nt          = X_kaggle.shape[0]                                                  # size of kaggle input
    preds       = []                                                                 # list of batched predictions
    num_batches = (nt // batch_size) + 1                                             # number of batches 

    for i in range(num_batches):                                                     # for each batch:
        _start  = i * batch_size                                                     # find index where batch starts
        _end    = np.minimum((i + 1) * batch_size, nt)                               # find index where batch ends 
        pred    = conv_apply(wts, X_kaggle[_start:_end,:,:,:], rng=mk)               # feedforward 
        preds.append(np.argmax(pred, axis=1))                                        # add index of highest prediction probability
    preds       = onp.array(np.concatenate(preds))                                   # turn it into an array

    f           = open(write_dir, 'w')                                               # write to file 
    f.write('id,label\n')
    for i in range(nt):
        _       = f.write(f'{i},{preds[i]}\n')
    f.close()

_X, _Y, _labels, _X_kaggle = load_data()
del _X, _Y, _labels
wts                        = load_it(os.path.join(os.getcwd(), 'wts_0.9106_256_165_0.0001_0.001_sgd'))
write_dir                  = os.path.join(os.getcwd(), 'submission.csv')
do_kaggle(write_dir, _X_kaggle, batch_size, conv_apply, wts, mk)

