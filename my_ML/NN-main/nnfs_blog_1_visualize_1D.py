# blog: Neural Network visualization

# this study is inspired by:
# http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/
# https://www.deep-teaching.org/notebooks/feed-forward-networks/exercise-simple-neural-network

# this study started by me trying to recreate the visualizations from Colah's blog, 
# starting with the one where the spirals become separated. luckily I found that 
# deep-teaching has a page which provides a first step: graphing some outputs with 
# a simple 2-2-1 network (in-hidden-out). let us be guided by some initial questions:

# 1. what can we visualize / graph? 
'''
the simple 2-2-1 network, for which we can use the XOR or two sines datasets, can be 
interpreted as a transformation f: R2 -> R2 which is the hidden layer, followed 
by a normal logistic regression / classification in the output layer, g: R2 -> R1. 

There are two ways to visualize (that i can think of so far):
1. from the viewpoint of the last layer, its input, no matter how crazy, is just a dataset. 
    we can visualize linear separability of the transformed data by plotting the last hidden 
    layer against classifier's logistic line [classifier pov]
    Q: is this the same as plotting transform in original space for 2-2-1 case?
2. from the viewpoint of the input axes, the dataset gets warped at each hidden layer.
    we can plot 1st hidden layer regression lines, their linear combination makes the 
    output decision boundary [if only 1 hidden layer] [original axes pov]
'''

# 2. what is the network actually doing?
'''
The first idea came to me with Colah's 1D donut example. This dataset needs 2 units in the 
hidden layer, for a very simple reason: each unit is a classifier, and we need one unit 
to fire when x > -0.5 and another to fire when x < 0.5. When both fire together we know 
we are in the intersection of their decision boundaries, or -0.5 < x < 0.5. But this 
precisely means we are identifying the red dataset! 

If these two neurons have positive weights going into the output layer, then that will be a 
positive classification. 

We can interpret neurons as classifiers working together to create subsets of the space
where they intersect. If the neurons' outputs have positive weights going into the next layer, 
their intersections will have the highest probability, while other areas have lower. 

One example to support this is the circle inside a donut dataset. Imagine trying to bound this circle 
using individual classifiers: we would need at least 3 lines, arranged so their intersection covers 
the circle. This implies there should be a solution using a 2-3-1 network. Creating such a 
network in the ConvNet.js website, we see that the solution happens to converge and it does in fact
look like a triangle! 

One interesting example exists: using 2 hidden units to create a decision boundary between two sines. 
This may appear to be using 3 classifiers according to how I described this "bounding process" above.
But it is in fact only 2 hidden units: one is horizontal, one is vertical (plot weights of individual neurons). 
What happens is the vertical one, imagined as a sigmoid function in 3D space above the x0, x1 axis (insert graph) 
adds its positive mass to the horizontal classifier in that half of the picture. That increases the 
horizontal classifier's value, in effect shifting it backward. In the middle we get a transition connecting these two. 
Together it looks like 3 classifier lines. This is a "hack" by the neural network to find a local minimum 
using the limited resources it has of 2 hidden units, similar to how a NN learned to hack pong by sending 
the ball into the top of the screen. 

This reminds me of function approximation in measure theory where we can approximate any function
using boxes, and if we need the approximation to be better, we use smaller boxes. This translates
directly to NNs because finer boxes can be obtained with more classifiers: more neurons in that layer.

Thus for toy examples we can figure out how many hidden units is the minimum (unless there is a hack)
by sketching a decision boundary using line segments. 

How back propagation finds weights to satisfy this solution is a different story ('create correlation' - trask)
I call my theory "bounding box theory" and it coincides with Trask because the classifiers will move in a way 
that bounds the dataset in a subset of space which will then "correlate" aka minimize loss wrt the cost function.
But there is no rule to how the classifiers move. Sometimes they converge to a triangle, other times not. 

Q: Do they just jump around randomly until a glimpse of "correlation" is found and latched on to? 
    (a vague triangle that incorporates most of red data)
    We can imagine all 3 lines having a similar slope and getting stuck in a local minimum because they can't bound the circle.

Q: we know 1 hidden layer with 2 neurons won't work, what about 2 hidden layers of 2 units? 3 hidden layers? 4? 

Q: this is all good and cool, but does this have anything to do with deep learning?
    I have yet to find an example that needs 2 hidden layers to solve. even mnist is solved by mlp
'''


import numpy as np
import matplotlib.pyplot as plt


def p(op, depth = 0): # print the comp graph
    line = '\t' * depth + repr(op)
    print(line)
    if op.typ != '_input':
        for v in op.ins:
            p(v, depth + 1)


def create_graph(op, g = []):   # topological sort 
    if op.typ == '_input':      # an input is a leaf node so we add it to the list and return
        if op not in g:         # so we don't have duplicates 
            g.append(op)
    else:                       # for an operator we need to iterate over its inputs
        for i in op.ins:
            g = create_graph(i, g)
        if op not in g:         # lastly, add the operator to the list itself
            g.append(op)
    return g


class Op:
    c = 0
    def __init__(self, ins, typ:str = ''):
        self.ins = ins 
        self.goes_to = []

        self.value = None
        self.grad = 0 # grad with respect to node's output

        self.typ = typ
        self.name = f'{typ}_{Op.c}'
        Op.c += 1

        for i in self.ins:
            i.goes_to.append(self)
    def forward():
        raise NotImplementedError()
    def backward():
        raise NotImplementedError()
    def __repr__(self):
        return f'{self.name}'

class Variable(Op):
    def __init__(self, v = None):
        super().__init__([], '_input')
        self.value = v
    def forward(self):
        return
    def backward(self):
        return

class Sigmoid(Op):
    def __init__(self, v1):
        super().__init__([v1], 'sigmoid')
    def forward(self):
        self.value = 1.0 / (1.0 + np.exp(-self.ins[0].value))
    def backward(self):
        self.ins[0].grad += self.grad * self.value * (1.0 - self.value)

class Tanh(Op):
    def __init__(self, v1):
        super().__init__([v1], 'tanh')
    def forward(self):
        a = np.exp(self.ins[0].value)
        b = np.exp(-self.ins[0].value)
        self.value = (a - b)/(a + b)
    def backward(self):
        self.ins[0].grad += self.grad * (1.0 - self.value * self.value)

class Relu(Op):
    def __init__(self, v1):
        super().__init__([v1], 'relu')
    def forward(self):
        self.value = np.maximum(0.0, self.ins[0])
    def backward(self):
        ret = np.zeros(self.value.shape)
        ret[np.where(self.ins[0] > 0)] = 1.0
        self.ins[0].grad += self.grad * ret

class MatMul(Op):
    def __init__(self, v1, v2):
        super().__init__([v1, v2], 'matmul')
    def forward(self):
        self.value = self.ins[0].value @ self.ins[1].value
    def backward(self):
        self.ins[0].grad += self.grad @ self.ins[1].value.T
        self.ins[1].grad += self.ins[0].value.T @ self.grad

class ScalarMul(Op): # this might be wrong - basiclaly hadamard prod
    def __init__(self, v1, v2):
        super().__init__([v1, v2], 'scalmul')
    def forward(self):
        self.value = self.ins[0].value * self.ins[1].value
    def backward(self):
        self.ins[0].grad += self.grad * self.ins[1].value
        self.ins[1].grad += self.grad * self.ins[0].value

class Add(Op): # stolen from sabinasz
    def __init__(self, v1, v2):
        super().__init__([v1, v2], 'add')
    def forward(self):
        self.value = self.ins[0].value + self.ins[1].value
    def backward(self):
        a = self.ins[0].value
        b = self.ins[1].value
        grad_wrt_a = self.grad
        while np.ndim(grad_wrt_a) > len(a.shape):
            grad_wrt_a = np.sum(grad_wrt_a, axis=0)
        for axis, size in enumerate(a.shape):
            if size == 1:
                grad_wrt_a = np.sum(grad_wrt_a, axis=axis, keepdims=True)
        grad_wrt_b = self.grad
        while np.ndim(grad_wrt_b) > len(b.shape):
            grad_wrt_b = np.sum(grad_wrt_b, axis=0)
        for axis, size in enumerate(b.shape):
            if size == 1:
                grad_wrt_b = np.sum(grad_wrt_b, axis=axis, keepdims=True)
        self.ins[0].grad += grad_wrt_a
        self.ins[1].grad += grad_wrt_b

class MSE(Op):
    def __init__(self, v1, v2):
        super().__init__([v1, v2], 'MSE_loss')
    def forward(self):
        self.error = self.ins[0].value - self.ins[1].value
        self.value = np.sum(np.square(self.error))
    def backward(self):
        self.ins[0].grad += self.error * self.grad
        self.ins[1].grad += -self.error * self.grad

class CrossEntropy(Op):
    def __init__(self, v1, v2):
        super().__init__([v1, v2], 'cross_entropy_loss')
    def forward(self):
        self.error = self.ins[0].value - self.ins[1].value
        self.value = np.mean(-self.ins[1].value * np.log(self.ins[0].value) - (1.0 - self.ins[1].value) * np.log(1.0 - self.ins[0].value))
    def backward(self):
        self.ins[0].grad += self.error * self.grad
        self.ins[1].grad += -self.error * self.grad





# 1D example

# generate data 
np.random.seed(0)
x_axis = np.arange(100)
red_data = np.stack([np.random.rand(100)*0.5 - 0.25, np.zeros(100)],1)
blue_data1 = np.stack([np.random.rand(50)*0.5 + .5, np.zeros(50)],1)
blue_data2 = np.stack([np.random.rand(50)*0.5 - 1., np.zeros(50)],1)
blu_data = np.append(blue_data1, blue_data2, 0)
labels = np.append(np.zeros(100), np.ones(100)).reshape([200,1])
X = np.append(blu_data[:,0], red_data[:,0]).reshape([200,1])



# one hidden layer - eta and epochs were changed until it worked
eta = 0.1
np.random.seed(0)
w0 = Variable(np.random.randn(1, 2)*0.1)
w1 = Variable(np.random.randn(2, 1)*0.1)
b0 = Variable(np.full((1,2),0.))
b1 = Variable(np.full((1,1),0.))
_input = Variable(X)
_labels = Variable(labels)
_layer1 = Sigmoid(Add(MatMul(_input, w0), b0))
_layer2 = Sigmoid(Add(MatMul(_layer1, w1), b1))
loss = CrossEntropy(_layer2, _labels)
_g = create_graph(loss, [])
for _e in range(1000):
    for i in _g: i.forward() 
    loss.grad = 1.0
    for i in reversed(_g): i.backward()
    w0.value -= eta * w0.grad 
    b0.value -= eta * b0.grad
    w1.value -= eta * w1.grad 
    b1.value -= eta * b1.grad
    for i in _g: i.grad = 0

# now the two hidden units are two classifiers. 
pred = _layer2.value
pred[pred > 0.5] = 1
pred[pred <= 0.5] = 0
np.sum(pred - labels) # ok we got 100% accuracy  



_xrange  = np.linspace(-1.0,1.0,1001)
_z1      = lambda x: x * w0.value[0,0] + b0.value[0,0] # preactivation 1
_z2      = lambda x: x * w0.value[0,1] + b0.value[0,1] # preactivation 2
_sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))          # activation
_h1      = lambda x: _sigmoid(_z1(x))                  # hidden unit 1 output
_h2      = lambda x: _sigmoid(_z2(x))                  # hidden unit 2 output



# plot 0: 1D decision boundary moving around 
_h1(-0.3856) # hidden unit 1 has decision bdry at x = -0.3856
_h2(0.3701)  # hidden unit 2 has decision bdry at x = 0.3701
plt.scatter(X[:100,0], np.arange(100), color = 'blue')
plt.scatter(X[100:,0], np.arange(100), color = 'red')
plt.vlines([-0.3856, 0.3701], ymin = 0, ymax=100)
plt.show()

# plot 1: hidden units as logistic regressions
plt.scatter(red_data[:,0], red_data[:,1], color='red')
plt.scatter(blu_data[:,0], blu_data[:,1], color='blue')
plt.plot(_xrange, _h1(_xrange), label = 'h1')
plt.plot(_xrange, _h2(_xrange), label = 'h2')
plt.legend()
plt.show()


# what does this graph tell us? a bit of a boolean logic puzzle. 
# h1 h2 out
#  1  1  0
#  0  1  1
#  0  0  0 
# If h2 has a weight of 1, then h1 would need a weight of -1 to make this work 
# let's check the weights of the output classifier:
w1.value
# they have a ratio of -1.
# plot 2: hidden units weighted for output 
plt.scatter(red_data[:,0], red_data[:,1], color='red')
plt.scatter(blu_data[:,0], blu_data[:,1], color='blue')
plt.plot(_xrange, w1.value[0] * _h1(_xrange), label = 'h1')
plt.plot(_xrange, w1.value[1] * _h2(_xrange), label = 'h2')
plt.legend()
plt.show()
# so the blue set occurs when these functions are both near 0 or both have the same high magnitude. 
# the red set occurs when only h2 is high while h1 is still 0. This shows how sigmoids are shifted 
# and re-weighted to get specific subsets of space that we want. 

# in the beginning I said imagine both hidden neurons fire for the red set. The weights our network
# learned here are different, possibly because it's more likely to find a solution to 
# h1 h2 out
#  1  1  0
#  0  1  1
#  0  0  0 
# than to:
# h1 h2 out
#  0  1  0
#  1  1  1
#  1  0  0 
# well let's try. We can introduce a 'prior' (heh) to the weights:
# w0 = Variable(np.array([[1.0,-1.0]]))
# which makes the sigmoids face each other so they are both 1 around the red set.
# the network learns fine. looking at the last table, we can imagine h1 and h2 both 
# have weight 1, then a bias of -1 can allow this to work. 
# the values i get for w1 are 8,8 and b1 = -12. pass these through a sigmoid and we get the table.


# since our hidden layer is a map from R1 to R2, we can view this transformation. 
# also, since the output layer is a logistic regression on this R2 input, we can see its line.
# plot 6: output as a logistic regression of its inputs
plt.plot(_h1(_xrange), _h2(_xrange))
plt.scatter(_h1(red_data[:,0]), _h2(red_data[:,0]), color='red')
plt.scatter(_h1(blu_data[:,0]), _h2(blu_data[:,0]), color='blue')
plt.plot(_xrange, (-b1.value[0,0] -w1.value[0,0] * _xrange) / w1.value[1,0])
plt.title('output as a logistic regression of its inputs')
plt.xlim(-.1,1.1)
plt.ylim(-.1,1.1)
plt.show()
# Thank you Colah, very cool. this deserves a training animation. 



# training loop for animation 
fig_dir = r'C:\Users\pwnag\Desktop\sup\deep_larn\my_plots'
_c = 0

eta = 0.1
np.random.seed(0)
w0 = Variable(np.random.randn(1, 2)*0.1)
w1 = Variable(np.random.randn(2, 1)*0.1)
b0 = Variable(np.full((1,2),0.))
b1 = Variable(np.full((1,1),0.))
_input = Variable(X)
_labels = Variable(labels)
_layer1 = Sigmoid(Add(MatMul(_input, w0), b0))
_layer2 = Sigmoid(Add(MatMul(_layer1, w1), b1))
loss = CrossEntropy(_layer2, _labels)
_g = create_graph(loss, [])
for _e in range(600): 
    for i in _g: i.forward() 
    loss.grad = 1.0
    for i in reversed(_g): i.backward()
    w0.value -= eta * w0.grad 
    b0.value -= eta * b0.grad
    w1.value -= eta * w1.grad 
    b1.value -= eta * b1.grad
    for i in _g: i.grad = 0

    if _c % 5 == 0:

        fig, axs = plt.subplots(2, 3, figsize = (10,6))

        axs[0, 0].scatter(red_data[:,0], red_data[:,1], color='red')
        axs[0, 0].scatter(blu_data[:,0], blu_data[:,1], color='blue')
        axs[0, 0].plot(_xrange, _h1(_xrange), label = 'h1')
        axs[0, 0].plot(_xrange, _h2(_xrange), label = 'h2')
        axs[0, 0].set_xlim(-1.1,1.1)
        axs[0, 0].set_ylim(-0.1,1.1)
        axs[0, 0].set_title('hidden units as logistic regressions')
        axs[0, 0].legend()

        axs[0, 1].scatter(red_data[:,0], red_data[:,1], color='red')
        axs[0, 1].scatter(blu_data[:,0], blu_data[:,1], color='blue')
        axs[0, 1].plot(_xrange, w1.value[0] * _h1(_xrange), label = 'h1')
        axs[0, 1].plot(_xrange, w1.value[1] * _h2(_xrange), label = 'h2')
        axs[0, 1].set_title('hidden units weighted')
        axs[0, 1].set_xlim(-1.1,1.1)
        axs[0, 1].legend()

        axs[0, 2].scatter(red_data[:,0], red_data[:,1], color='red')
        axs[0, 2].scatter(blu_data[:,0], blu_data[:,1], color='blue')
        axs[0, 2].plot(_xrange, w1.value[0] * _h1(_xrange) + w1.value[1] * _h2(_xrange))
        axs[0, 2].set_title('sum of weighted hidden units')
        axs[0, 2].set_xlim(-1.1,1.1)

        axs[1, 0].scatter(red_data[:,0], red_data[:,1], color='red')
        axs[1, 0].scatter(blu_data[:,0], blu_data[:,1], color='blue')
        axs[1, 0].plot(_xrange, w1.value[0] * _h1(_xrange) + w1.value[1] * _h2(_xrange) + b1.value[0,0])
        axs[1, 0].set_title('add bias')
        axs[1, 0].set_xlim(-1.1,1.1)
        axs[1, 0].set_ylim(-4.9,5.1)

        axs[1, 1].scatter(red_data[:,0], red_data[:,1], color='red')
        axs[1, 1].scatter(blu_data[:,0], blu_data[:,1], color='blue')
        axs[1, 1].plot(_xrange, _sigmoid(w1.value[0] * _h1(_xrange) + w1.value[1] * _h2(_xrange) + b1.value[0,0]))
        axs[1, 1].set_title('output')
        axs[1, 1].set_xlim(-1.1,1.1)
        axs[1, 1].set_ylim(-0.1,1.1)

        axs[1, 2].plot(_h1(_xrange), _h2(_xrange))
        axs[1, 2].scatter(_h1(red_data[:,0]), _h2(red_data[:,0]), color='red')
        axs[1, 2].scatter(_h1(blu_data[:,0]), _h2(blu_data[:,0]), color='blue')
        axs[1, 2].plot(_xrange, (-b1.value[0,0] -w1.value[0,0] * _xrange) / w1.value[1,0])
        axs[1, 2].set_title('output as logistic regression of its inputs')
        axs[1, 2].set_xlim(-0.1,1.1)
        axs[1, 2].set_ylim(-0.1,1.1)

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        #plt.show()

        counter = str(_c).rjust(5, '0')
        plt.savefig(fig_dir + f'\\{counter}.png', bbox_inches='tight')
        plt.close('all')
    _c += 1 


import os
import imageio

images = []
for file_name in os.listdir(fig_dir):
    file_path = os.path.join(fig_dir, file_name)
    images.append(imageio.imread(file_path))
imageio.mimsave(fig_dir + '\\movie.gif', images, duration = 0.05)


# what does this animation tell us? Well it reveals the connection between h1, h2 and the final logistic regression.
# let's think of h1 and h2 as the new transformed coordinates (which they are). h1 says points below -0.5 get a 1, 
# the rest get a 0 on the h1 scale. 
# on the other hand, h2 says points below 0.5 get a 1, the rest get a 0 on the h2 scale.

# together, they made the left blue points stay around 0,0 in h1,h2 space. the red points get a 0 on h1, 1 on h2 scale, 
# so they go to the top left corner. the right blue part gets a high value for h0, h1 and is sent to the top right corner.

# we should ignore the curvature of this line because it makes everything look more crazy and think of the sigmoids 
# as smooth versions of step functions. 
# so we have sent the points to the 3 corners, and it so happens that now they are linearly separable. 

# the part that makes this unique is of course the coordinate transformations. 
# we used step functions (sigmoids) to partition space with different combinations of the step functions' values: 0, 1. 
# for example left blue got 1,1 ;; red got 0,1 ;; and right blue got 0,0. 

# wherever there should be a boundary between different data classes, put one classifier there. 
# each classifier cuts R in two parts. Each subset of interest then gets a unique vector of the sigmoids' values:
# h1 h2 out
#  1  1  0  <- left blue
#  0  1  1  <- red
#  0  0  0  <- right blue

# a classifier's decision boundary can be thought of as an IF statement. 
# h1 says if x < -0.5 ;; h2 says if x < 0.5. 
# a linear combination of IF statements can then allow us to create intersections with specific numerical values.
#     for example, 1 * h1 + 1 * h2 will have value 2 for the set (x < -0.5 AND x < 0.5)(left blue), 
#     1 for the set (-0.5 < x AND x < 0.5)(red set), 0 for the set (-0.5 < x AND 0.5 < x)(right blue).
#     but using -1 * h1 + 1 * h2 will have value 0 for left blue, 1 for red, 0 for right blue, which coincides with our labels.
# as Trask calls it, this is 'selective correlation'. we selected, and weighted, subsets so they correlate with our labels.
# that is why I called h1, h2 IF statements: because select some part of space, and ignore the rest, thus forming a subset.

# * * * since every subset of interest gets its own unique vector of the sigmoid's values (0,1), this corresponds to a unique point
# on an integer lattice (or rather, a hypercube since we only have 0,1 and as many dimensions as hidden units). 
# this hypercube of points is the input to the next layer. 
# either these points are linearly separable, or they will require another hidden layer to realign. 
# in our example we see a square (2D because 2 hidden units) with the 3 subsets sent to the 3 corners: (0,0), (0,1), (1,1).

# ok time to wrap this up. in summary:
# 1. a hidden unit cuts its input space into 0,1. 
# 2. we get subsets where the cuts of several hidden units overlap. each subset has a unique vector of the sigmoids' values. 
# 3. each subset is sent to a unique point on a hypercube corresponding to its vector. This cube is the input to the next layer. 
#     by some layer this cube's vertices should become linearly separable. 
# 4. the subsets' vectors make up a boolean table. 
# 5. the boolean table becomes a system of linear equations. we hope to 'create correlation' with the output, by adjusting weights.


# this study has answered a lot of questions and opened up many more. 
# 1. XOR can be solved easily now because a linear transformation can collapse the 2D space along the line y=x and we are left with this 1D example.
# 2. we used sigmoid. tanh is similar since it partitions space into -1,1. 
#     relu seems fundamentally different - how can it be interpreted? (partitions into 0,x) or (collapses space to 0 where negative)
# 3. can i create a dataset that is just as minimal but requires 2 hidden layers?



# next example: 2 hidden layers, this dataset is R-B-R-B-R 
# the hypothesis is that if we can get a bent line like in the 1D example in hidden layer 1, 
# then two regressions can surround the blue subsets for the next layer to separate them. 

# but i will do this after the xor, sine examples since they're simpler. 


np.random.seed(0)
x_axis = np.arange(100)
red_data1 = np.stack([np.random.rand(40)*0.2 - 0.1, np.zeros(40)],1)
red_data2 = np.stack([np.random.rand(30)*0.2 + 0.5, np.zeros(30)],1)
red_data3 = np.stack([np.random.rand(30)*0.2 - 0.7, np.zeros(30)],1)
red_data = np.append(np.append(red_data1, red_data2, 0), red_data3, 0)
blue_data1 = np.stack([np.random.rand(50)*0.2 + .2, np.zeros(50)],1)
blue_data2 = np.stack([np.random.rand(50)*0.2 - .4, np.zeros(50)],1)
blu_data = np.append(blue_data1, blue_data2, 0)
labels = np.append(np.zeros(100), np.ones(100)).reshape([200,1])
X = np.append(blu_data[:,0], red_data[:,0]).reshape([200,1])
plt.plot(X)
plt.show()



# two hidden layers
eta = 0.1
np.random.seed(0)
w0 = Variable(np.random.randn(1, 2)*0.1)
w1 = Variable(np.random.randn(2, 2)*0.1)
w2 = Variable(np.random.randn(2, 1)*0.1)
b0 = Variable(np.full((1,2),0.))
b1 = Variable(np.full((1,2),0.))
b2 = Variable(np.full((1,1),0.))
_input = Variable(X)
_labels = Variable(labels)
_layer1 = Sigmoid(Add(MatMul(_input, w0), b0))
_layer2 = Sigmoid(Add(MatMul(_layer1, w1), b1))
_layer3 = Sigmoid(Add(MatMul(_layer2, w2), b2))
loss = CrossEntropy(_layer3, _labels)
_g = create_graph(loss, [])
for _e in range(20_000):
    for i in _g: i.forward() 
    loss.grad = 1.0
    for i in reversed(_g): i.backward()
    w0.value -= eta * w0.grad 
    b0.value -= eta * b0.grad
    w1.value -= eta * w1.grad 
    b1.value -= eta * b1.grad
    w2.value -= eta * w2.grad 
    b2.value -= eta * b2.grad
    for i in _g: i.grad = 0

# now the two hidden units are two classifiers. 
pred = _layer3.value
pred[pred > 0.5] = 1
pred[pred <= 0.5] = 0
np.sum(pred - labels) # ok we got 100% accuracy  



_xrange  = np.linspace(-1.0,1.0,1001)
_sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))          # activation
_z11      = lambda x: x * w0.value[0,0] + b0.value[0,0] # preactivation 1
_z12      = lambda x: x * w0.value[0,1] + b0.value[0,1] # preactivation 2
_h11      = lambda x: _sigmoid(_z11(x))                  # hidden unit 1 output
_h12      = lambda x: _sigmoid(_z12(x))                  # hidden unit 2 output

_z21      = lambda x: x * w1.value[0,0] + b1.value[0,0] # preactivation 1
_z22      = lambda x: x * w1.value[0,1] + b1.value[0,1] # preactivation 2
_h21      = lambda x: _sigmoid(_z21(x))                  # hidden unit 1 output
_h22      = lambda x: _sigmoid(_z22(x))                  # hidden unit 2 output



plt.plot(_h11(_xrange), _h12(_xrange))
plt.scatter(_h11(red_data[:,0]), _h12(red_data[:,0]), color='red')
plt.scatter(_h11(blu_data[:,0]), _h12(blu_data[:,0]), color='blue')
plt.title('output as a logistic regression of its inputs')
plt.xlim(-.1,1.1)
plt.ylim(-.1,1.1)
plt.show()

plt.scatter(_h21(_h11(red_data[:,0])), _h22(_h12(red_data[:,0])), color='red')
plt.scatter(_h21(_h11(blu_data[:,0])), _h22(_h12(blu_data[:,0])), color='blue')
plt.plot(_xrange, (-b2.value[0,0] -w2.value[0,0] * _xrange) / w2.value[1,0])
plt.title('output as a logistic regression of its inputs')
#plt.xlim(-.1,1.1)
#plt.ylim(-.1,1.1)
plt.show()