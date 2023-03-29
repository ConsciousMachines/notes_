
# - - - how to install old shit 
# this script uses pythoon 2.7 since it has from __future__ import print_function
# 1. conda create -n tf_old python=2.7 ipython
# download oldest tensorflow from 2017, for python 2.7
# 2. https://pypi.org/project/tensorflow/1.4.1/#files
# downgrade the microsoft python extension to work with python 2.7
# 3. on the extension page, under remove, click "download another version" and get v2021.9.1246542782.
# install pillow, in the actiavted env. 
# 4. conda activate tf_old
#    pip install pillow
# for tensorboard, from the deep-dram directory, run:
# 5. mkdir tb_graph
#    /home/chad/miniconda3/envs/tf_old/bin/tensorboard --logdir tb_graph



# music
# https://www.youtube.com/watch?v=AlPpnvLbWBE&list=RDHhkGh2cqvE0&index=16
# https://www.youtube.com/watch?v=tADuImz6pJk
# https://www.youtube.com/watch?v=xHbSnT5MNdU
# https://www.youtube.com/watch?v=fD6aIZhFDJU&list=RDGMEMYH9CUrFO7CfLJpaD7UR85w&index=2
# https://youtu.be/ZZIDRnErsPc?list=RDGMEMYH9CUrFO7CfLJpaD7UR85w





# Showing the lap_normalize graph with TensorBoard
#lap_graph = tf.Graph()
#with lap_graph.as_default():
#    lap_in = tf.placeholder(np.float32, name='lap_in')
#    lap_out = lap_normalize(lap_in)



# https://github.com/bapoczos/deep-dream-tensorflow/blob/master/deepdream.ipynb


import os
import sys 
import numpy as np
import PIL.Image as im
import tensorflow as tf

out_dir = r'/home/chad/Desktop/_backups/notes/my_ML/deep_dram'
model_fn = os.path.join(out_dir, r'/home/chad/Desktop/_backups/notes/ignore/inception5h/tensorflow_inception_graph.pb')
os.chdir(out_dir)

# creating TensorFlow session and loading the model from the model_fn file 
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input') # define the input tensor
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})

layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]
print('Number of layers', len(layers))
print('Total number of feature channels:', sum(feature_nums))

def T(layer): # Helper for getting layer output tensor
    return graph.get_tensor_by_name("import/%s:0"%layer)

layer = layers[4]
print(layer)
layer = layer.split("/")[1]
print(layer)
T(layer)

for l, layer in enumerate(layers):
    layer = layer.split("/")[1]
    num_channels = T(layer).get_shape()
    print(layer, num_channels)

def showarray(a): # create a jpeg file from an array a and visualize it
    a = np.uint8(np.clip(a, 0, 1) * 255) # clip the values to be between 0 and 255
    im.fromarray(a).show()
    
def visstd(a, s = 0.1): # Normalize the image range for visualization
    return (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5 # i think this is an arbitrary way to put array in the range 0,1

pops = graph.get_operations()[:10]
for i, op in enumerate(pops):
    print(i, op.name)



# T E N S O R B O A R D 

# https://stackoverflow.com/questions/42003846/retraining-inception5h-model-from-tensorflow-android-camera-demo
# https://github.com/googlecodelabs/tensorflow-for-poets-2/blob/master/scripts/graph_pb2tb.py

def graph_to_tensorboard(graph, out_dir):
  with tf.Session():
    train_writer = tf.summary.FileWriter(out_dir)
    train_writer.add_graph(graph)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
graph_to_tensorboard(graph, 'tb_graph/inception5h')
  


























#   R E N D E R   N A I V E 
# Picking some internal layer. Note that we use outputs before applying the ReLU nonlinearity
# to have non-zero gradients for features with negative initial activations.
np.random.seed(0)
img_noise = np.array(np.random.rand(224,224,3) + 100.0).astype(np.float32)
img = img_noise.copy()
layer                   = 'mixed4d_3x3_bottleneck_pre_relu'

t_obj                   = T(layer)[:,:,:,139]         # the feature maps where we want to maximize the activations of the neurons
_loss                   = tf.reduce_mean(t_obj)           # this is our optimization objective - mean of neuron activations in t_obj
t_grad                  = tf.gradients(_loss, t_input)[0] # calculate the gradient of the objective function!!!

for i in range(20): 
    g, score            = sess.run([t_grad, _loss], {t_input:img})
    g                  /= g.std() + 1e-8    # normalizing the gradient, so the same step size should work for different layers and networks
    img                += g                 # update
showarray(visstd(img))






#   M U L T I S C A L E 

def tffunc(*argtypes): # basically some weird implementation specific stuff that i won't bother with
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.placeholder, argtypes)) # create a list of placeholders w specific types
    def wrap(f):
        out = f(*placeholders) # apply given function (resize) to placeholders
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap
def resize(img, size): # Helper function that uses TF to resize an image
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)

#if False:
    
t_obj                               = T(layer)[:,:,:,139]
img                                 = img_noise.copy()
octave_scale                        = 1.4

_loss                               = tf.reduce_mean(t_obj) # defining the optimization objective
t_grad                              = tf.gradients(_loss, t_input)[0] # behold the power of automatic differentiation!

for octave in range(3): # number of octaves
    if octave > 0:      # calc new height & width when scaling up by octave_sclae
        hw                          = np.float32(img.shape[:2]) * octave_scale
        img                         = resize(img, np.int32(hw)) # rescale the image to the new size
    for i in range(10):
        # compute grad of image by taking grad of small tiles. apply random shifts to image to blur boundaries between iterations
        sz                          = 512 # tile size
        h, w                        = img.shape[:2] # size of the image
        sx, sy                      = np.random.randint(sz, size=2) # random shift numbers generated
        img_shift                   = np.roll(np.roll(img, sx, 1), sy, 0) #shift the whole image. np.roll = Roll array elements along a given axis
        grad                        = np.zeros_like(img)

        for y in range(0, max(h - sz//2, sz), sz):     # NOTE: this leaves patches of the image un-gradiented.
            for x in range(0, max(w - sz//2, sz), sz): # alternative is to pad (then waste grad comp?) also the net takes in diff dims.
                sub                 = img_shift[y:y + sz,x:x + sz]           # tile of varying size: <= 512
                g                   = sess.run(t_grad, {t_input:sub})        # grad of tile
                grad[y:y+sz,x:x+sz] = g                                      # assemble real grad from tile
        g                           = np.roll(np.roll(grad, -sx, 1), -sy, 0) # unroll
        g                          /= g.std() + 1e-8 # normalizing the gradient, so the same step size should work for different layers and networks
        img                        += g # update 
showarray(visstd(img))





#   L A P L A C I A N
# OK the point of this is to decompose the gradient (image) into hi-freq components at various levels, and normalize them
# then add them back to the image. 
# https://github.com/mzhao98/laplacian_blend
# ok so basically the laplace algorithm goes like this: 
# 1. we take an image, convolve with Gaussian and downsample (gaus to avoid alias)
# 2. we upsample the downsampled image by adding 0s, and apply Gaus filter.
# 3. we subtract the original image with the upsampled to get "hi freq noise" which is Laplace part.
# 4. repeat the above step 3 times.
# 5. apply Gaus to the mask at the 3 scales, multiply by the 3 Laplacians. 
# 6. take the smallest image, upsample, multiply by mask, and add the finished noise. 
# 7. upsample it again, add the 2nd noise. then repeat. 
# Laplacian pyramid: Used to reconstruct an upsampled image from an image lower in the pyramid (with less resolution)
# to upsize, Perform a convolution with the same kernel (multiplied by 4) to approximate the values of the "missing pixels"
# the entire pyramid can be used to perfeclty reconstruct the image. https://pyrtools.readthedocs.io/en/latest/tutorials/tutorial2.html

k                         = np.float32([1,4,6,4,1])                                           # used to make Gaus kernel
k                         = np.outer(k, k)                                                    # Gaus kernel
k5x5                      = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)              # some sort of Conv-Transpose Gaussian kernel. 

def normalize_std(img):                                                            # Normalize image by making its standard deviation = 1.0
    with tf.name_scope('normalize'):
        std               = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img / tf.maximum(std, 1e-10)

# build the Laplacian pyramid normalization graph
lap_ph                    = tf.placeholder(np.float32)                                        # just a placeholder
img                       = tf.expand_dims(lap_ph,0)                                          # exapnd dim for conv

# Build Laplacian pyramid with 4 splits
levels                    = []
for i in range(4):
    # Split the image into lo and hi frequency components
    with tf.name_scope('split'):
        lo                = tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')                        # just conv with Gaussian kernel
        lo2               = tf.nn.conv2d_transpose(lo, k5x5*4, tf.shape(img), [1,2,2,1])      # some form of upsampling with Gaus kernel.
        hi                = img - lo2                                                         # the hi-freq noise, Laplacian part
    levels.append(hi)                                                                         # add Laplacian part to list
    img                   = lo                                                                # on next iteration, do the same thing to the downsampled image.
levels.append(img)                                                                            # at the end we append the last downsampled image. 
tlevels                   = [normalize_std(i) for i in levels[::-1]]                          # reverse order of downsampled imgs, and normalize
# Merge Laplacian pyramid
out                       = tlevels[0]                                                        # start with the smallest downsampled image
for hi in tlevels[1:]:                                                                        # loop over the Laplacian hi-freq components
    with tf.name_scope('merge'):
        out               = tf.nn.conv2d_transpose(out, k5x5*4, tf.shape(hi), [1,2,2,1]) + hi # upscale the image and add the noise.
out                       = out[0,:,:,:]                                                      # unbatch





t_obj                               = T(layer)[:,:,:,139]
octave_scale                        = 2.0
t_score                             = tf.reduce_mean(t_obj) # defining the optimization objective
t_grad                              = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!

img                                 = img_noise.copy()
for octave in range(3):
    if octave > 0:
        hw                          = np.float32(img.shape[:2]) * octave_scale
        img                         = resize(img, np.int32(hw))
    for i in range(10):
        # G R A D   T I L E D   F N 
        sz                          = 512
        h, w                        = img.shape[:2]
        sx, sy                      = np.random.randint(sz, size=2) 
        img_shift                   = np.roll(np.roll(img, sx, 1), sy, 0)
        grad                        = np.zeros_like(img)
        for y in range(0, max(h-sz//2, sz),sz):
            for x in range(0, max(w-sz//2, sz),sz):
                sub                 = img_shift[y:y+sz,x:x+sz] 
                g                   = sess.run(t_grad, {t_input:sub}) 
                grad[y:y+sz,x:x+sz] = g 
        g                           = np.roll(np.roll(grad, -sx, 1), -sy, 0) 
        g                           = out.eval({lap_ph:g}) # the only different line if g /= g.std() is replaced by lap_norm_func
        img                        += g
showarray(visstd(img))





#   D E E P   D R E A M

img0                                = np.float32(im.open(os.path.join(out_dir, 'android.png')))
t_obj                               = T(layer)[:,:,:,66]
step                                = 1.5
octave_n                            = 2
octave_scale                        = 1.1

t_score                             = tf.reduce_mean(t_obj) # defining the optimization objective
t_grad                              = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!

# split the image into a number of octaves getting smaller and smaller images
img                                 = img0
octaves                             = []
for i in range(octave_n-1):
    hw                              = img.shape[:2] #image height and width
    lo                              = resize(img, np.int32(np.float32(hw)/octave_scale)) #low frequency parts (smaller image)
    hi                              = img - resize(lo, hw) #high frequency parts (details)
    img                             = lo # next iteration rescale this one
    octaves.append(hi) # add the details to octaves

# generate details octave by octave from samll image to large
for octave in range(octave_n):
    if octave > 0:
        hi                          = octaves[-octave]
        img                         = resize(img, hi.shape[:2]) + hi
    for i in range(10):
        sz                          = 512
        h, w                        = img.shape[:2] # size of the image
        sx, sy                      = np.random.randint(sz, size=2) # random shift numbers generated
        img_shift                   = np.roll(np.roll(img, sx, 1), sy, 0) #shift the whole image. np.roll = Roll array elements along a given axis
        grad                        = np.zeros_like(img)
        for y in range(0, max(h-sz//2, sz),sz):
            for x in range(0, max(w-sz//2, sz),sz):
                sub                 = img_shift[y:y+sz,x:x+sz] # get the image patch (tile)
                g                   = sess.run(t_grad, {t_input:sub}) # calculate the gradient only in the image patch not in the whole image!
                grad[y:y+sz,x:x+sz] = g # put the whole gradient together from the tiled gradients g
        g                           = np.roll(np.roll(grad, -sx, 1), -sy, 0) # shift back
        img                        += g * (step / (np.abs(g).mean()+1e-7)) # TODO: replace this part with laplacian norm
showarray(img / 255.0)

# so basically this deep dream algo gave up on the laplacian normalization. 

