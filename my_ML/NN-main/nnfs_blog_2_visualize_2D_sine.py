import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

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




# 2D example - sines (not normalized)
VERT_OFFSET = 1.5
_bigrange = np.linspace(0,np.pi* 2.4, 1001)
class1func = np.stack([_bigrange, np.sin(_bigrange)], 1)
class2func = np.stack([_bigrange, np.sin(_bigrange) + VERT_OFFSET], 1)
_xrange = np.linspace(0,np.pi* 2.4, 50)
class1 = np.stack([_xrange, np.sin(_xrange)], 1)               # class 1 is bottom sine
class2 = np.stack([_xrange, np.sin(_xrange) + VERT_OFFSET], 1) # class 2 is top sine
X = np.append(class1, class2, 0)
labels = np.append(np.zeros(50), np.ones(50), 0).reshape([100,1])
#plt.plot(_bigrange, class1func[:,1], color='blue', label = 'class 1')
#plt.plot(_bigrange, class2func[:,1], color='orange', label = 'class 2')
#plt.scatter(_xrange, class1[:,1], color='blue', label='class 1 points')
#plt.scatter(_xrange, class2[:,1], color='orange', label='class 2 points')
#plt.legend()
#plt.show()


# one hidden layer 
eta = 0.1
np.random.seed(0)
w0 = Variable(np.random.randn(2, 2)*0.1)
w1 = Variable(np.random.randn(2, 1)*0.1)
b0 = Variable(np.full((1,2),0.))
b1 = Variable(np.full((1,1),0.))
_input = Variable(X)
_labels = Variable(labels)
_layer1 = Tanh(Add(MatMul(_input, w0), b0))
_layer2 = Sigmoid(Add(MatMul(_layer1, w1), b1))
loss = CrossEntropy(_layer2, _labels)
_g = create_graph(loss, [])
for _e in range(3_000):
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
np.sum(np.abs(pred - labels)) # ok we got 100% accuracy  


# P l O T T I N G   S T U F F
# create a square of points in 2D, for plotting decision boundary of area
grid_density = 100 
x1 = np.linspace(0,np.pi* 2.4,grid_density)
x2 = np.linspace(-1,2.5,grid_density)
mash = np.meshgrid(x1,x2)
_points = np.ndarray((grid_density**2, 2))
_points[:,0] = mash[0].flatten()
_points[:,1] = mash[1].flatten()
# needed for plotting decision boundary
def forward_pass(_x):
    _input.value = _x 
    for i in _g[:-2]: i.forward()
    return _g[4].value, _g[5].value, _g[9].value, _g[10].value
# generate decision boundary points
Z1, A1, Z2, A2 = forward_pass(_points)
# used for grid
_soyrange = np.linspace(-2.0, 2.0, 1000)
# horizontal + vertical grid
gryd = []
for ii in range(-20,20,1):
    gryd.append(np.stack([np.zeros(_soyrange.shape) + (ii)*0.1, _soyrange], 1))
    gryd.append(np.stack([_soyrange, np.zeros(_soyrange.shape) + (ii)*0.1], 1))
# test grid
#for g in gryd:
#    plt.plot(g[:,0], g[:,1], color = 'gray', linewidth=0.5)
#plt.show()



_sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))      # activation
def _tanh(_x):
    a = np.exp(_x)
    b = np.exp(-_x)
    return (a - b)/(a + b)
_Z1 = lambda x: x @ w0.value + b0.value            # pre-activations of layer 1
_A1 = lambda x: _tanh(_Z1(x))                   # activations of layer 1




# plot 0: heatmaps of hidden units' logistic regressions, to get 0,1 values of quadrants
hidden_sigmoids = A1[:,0].reshape([100,100]) + A1[:,1].reshape([100,100])
figure, axes = plt.subplots()
axes.set_title('hidden units as logistic regressions')
c = axes.pcolormesh(mash[0], mash[1], hidden_sigmoids, cmap='cool_r', vmin=0.0, vmax=2.0)
figure.colorbar(c)
# the two logistic lines
axes.plot(_xrange, (-b0.value[0,0] -_xrange * w0.value[0,0]) / w0.value[1,0], label='h1', color='blue')
axes.plot(_xrange, (-b0.value[0,1] -_xrange * w0.value[0,1]) / w0.value[1,1], label='h2', color='orange')
axes.legend()
# the sine points
axes.scatter(_xrange, class1[:,1], color='blue')
axes.scatter(_xrange, class2[:,1], color='orange')
plt.show()

# we can already see the bounding box theory in effect: the two classifiers
# found a way to bound one class, the other class gets "the rest" of the area
# from this plot we get the following boolean table:
# h1 h2 out
#  0  0  0  <- left quadrant
#  0  1  0  <- bottom quadrant
#  1  0  1  <- top quadrant, class 2
#  1  1  0  <- right quadrant
# this tells us that the orange class will be mapped to point (1,0) in h1,h2 space. 


# plot 4: logistic regression of the hidden units
l1 = _A1(X)
func1 = _A1(class1func)
func2 = _A1(class2func)
plt.plot(func1[:,0], func1[:,1], color = 'blue') # transform of the entire function 1
plt.plot(func2[:,0], func2[:,1], color = 'orange')   # transform of the entire function 2
plt.scatter(l1[:50,0], l1[:50,1], color='blue')  # transform of the points 1
plt.scatter(l1[50:,0], l1[50:,1], color='orange')    # transform of the points 2
plt.plot(_xrange, (-b1.value[0,0] -w1.value[0,0] * _xrange) / w1.value[1,0]) # logistic reg line
plt.title('output as logistic regression of hiddens')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)
plt.show()

# once again we see subsets being mapped to cube corners. 
# we have 4 subsets corresponding to the 4 quadrants made by the two logistic reg lines of layer 1.
# the top intersection of the regressions covers the top sine, "the rest" covers the bottom sine.


# UNNECESSARY plot -1: hidden units 1,2 as logistic regressions, and their combination (output's decision boundary) 
'''
preds = A2.squeeze()
_nos = _points[preds < 0.5]  # the NO points
_yes = _points[preds >= 0.5] # the YES points
# the grid corresponding to final output
plt.scatter(_nos[:,0],_nos[:,1], alpha=1.0, marker='s', color="#aaccee")
plt.scatter(_yes[:,0],_yes[:,1], alpha=1.0, marker='s', color="#eeccaa")
# the sine points
plt.scatter(_xrange, class1[:,1], color='blue')
plt.scatter(_xrange, class2[:,1], color='orange')
plt.show()
'''


# training loop for animation 
fig_dir = r'C:\Users\pwnag\Desktop\sup\deep_larn\my_plots'
_c = 0

eta = 0.1
np.random.seed(0)
w0 = Variable(np.random.randn(2, 2)*0.1)
w1 = Variable(np.random.randn(2, 1)*0.1)
b0 = Variable(np.full((1,2),0.))
b1 = Variable(np.full((1,1),0.))
_input = Variable(X)
_labels = Variable(labels)
_layer1 = Sigmoid(Add(MatMul(_input, w0), b0))
_layer2 = Sigmoid(Add(MatMul(_layer1, w1), b1))
loss = CrossEntropy(_layer2, _labels)
_g = create_graph(loss, [])
for _e in range(10_000):
    _input.value = X
    for i in _g: i.forward() 
    loss.grad = 1.0
    for i in reversed(_g): i.backward()
    w0.value -= eta * w0.grad 
    b0.value -= eta * b0.grad
    w1.value -= eta * w1.grad 
    b1.value -= eta * b1.grad
    for i in _g: i.grad = 0

    if _c < 200: #_c % 50 == 0:


        fig, axs = plt.subplots(2, 3, figsize = (12,8))

        # plot 0: heatmaps of hidden units' logistic regressions, to get 0,1 values of quadrants
        Z1, A1, Z2, A2 = forward_pass(_points)
        hidden_sigmoids = A1[:,0].reshape([100,100]) + A1[:,1].reshape([100,100])
        axs[0, 0].set_title('hidden units as logistic regressions')
        c = axs[0, 0].pcolormesh(mash[0], mash[1], hidden_sigmoids, cmap='cool_r', vmin=0.0, vmax=2.0)
        #fig.colorbar(c)
        # the two logistic lines
        axs[0, 0].plot(_xrange, (-b0.value[0,0] -_xrange * w0.value[0,0]) / w0.value[1,0], label='h1', color='blue')
        axs[0, 0].plot(_xrange, (-b0.value[0,1] -_xrange * w0.value[0,1]) / w0.value[1,1], label='h2', color='orange')
        axs[0, 0].legend()
        # the sine points
        axs[0, 0].scatter(_xrange, class1[:,1], color='blue')
        axs[0, 0].scatter(_xrange, class2[:,1], color='orange')
        axs[0, 0].set_ylim(-1.0,2.5)



        # plot 1: hidden units weighted
        hidden_sigmoids = A1[:,0].reshape([100,100]) * w1.value[0,0] + A1[:,1].reshape([100,100]) * w1.value[1,0]
        axs[0, 1].set_title('hidden units weighted')
        c = axs[0, 1].pcolormesh(mash[0], mash[1], hidden_sigmoids, cmap='cool_r')
        # the two logistic lines
        axs[0, 1].plot(_xrange, (-b0.value[0,0] -_xrange * w0.value[0,0]) / w0.value[1,0], label='h1', color='blue')
        axs[0, 1].plot(_xrange, (-b0.value[0,1] -_xrange * w0.value[0,1]) / w0.value[1,1], label='h2', color='orange')
        axs[0, 1].legend()
        # the sine points
        axs[0, 1].scatter(_xrange, class1[:,1], color='blue')
        axs[0, 1].scatter(_xrange, class2[:,1], color='orange')
        axs[0, 1].set_ylim(-1.0,2.5)



        # plot 2: add bias
        hidden_sigmoids = A1[:,0].reshape([100,100]) * w1.value[0,0] + A1[:,1].reshape([100,100]) * w1.value[1,0] + b1.value[0,0]
        axs[0, 2].set_title('add bias')
        c = axs[0, 2].pcolormesh(mash[0], mash[1], hidden_sigmoids, cmap='cool_r')
        # the two logistic lines
        axs[0, 2].plot(_xrange, (-b0.value[0,0] -_xrange * w0.value[0,0]) / w0.value[1,0], label='h1', color='blue')
        axs[0, 2].plot(_xrange, (-b0.value[0,1] -_xrange * w0.value[0,1]) / w0.value[1,1], label='h2', color='orange')
        axs[0, 2].legend()
        # the sine points
        axs[0, 2].scatter(_xrange, class1[:,1], color='blue')
        axs[0, 2].scatter(_xrange, class2[:,1], color='orange')
        axs[0, 2].set_ylim(-1.0,2.5)



        # plot 3: output
        axs[1, 0].set_title('output')
        c = axs[1, 0].pcolormesh(mash[0], mash[1], A2.reshape([100,100]), cmap='cool_r')
        # the two logistic lines
        axs[1, 0].plot(_xrange, (-b0.value[0,0] -_xrange * w0.value[0,0]) / w0.value[1,0], label='h1', color='blue')
        axs[1, 0].plot(_xrange, (-b0.value[0,1] -_xrange * w0.value[0,1]) / w0.value[1,1], label='h2', color='orange')
        axs[1, 0].legend()
        # the sine points
        axs[1, 0].scatter(_xrange, class1[:,1], color='blue')
        axs[1, 0].scatter(_xrange, class2[:,1], color='orange')
        axs[1, 0].set_ylim(-1.0,2.5)



        # plot 4: logistic regression of the hidden units
        l1 = _A1(X)
        func1 = _A1(class1func)
        func2 = _A1(class2func)
        axs[1, 1].plot(func1[:,0], func1[:,1], color = 'blue') # transform of the entire function 1
        axs[1, 1].plot(func2[:,0], func2[:,1], color = 'orange')   # transform of the entire function 2
        axs[1, 1].scatter(l1[:50,0], l1[:50,1], color='blue')  # transform of the points 1
        axs[1, 1].scatter(l1[50:,0], l1[50:,1], color='orange')    # transform of the points 2
        axs[1, 1].plot(_xrange, (-b1.value[0,0] -w1.value[0,0] * _xrange) / w1.value[1,0]) # logistic reg line
        axs[1, 1].set_title('output as logistic regression of hiddens')
        axs[1, 1].set_xlim(-0.1,1.1)
        axs[1, 1].set_ylim(-0.1,1.1)


        # plot 5: visualize space transformation by the 2 hidden units
        l1 = _A1(X)
        func1 = _A1(class1func)
        func2 = _A1(class2func)
        # horizontal + vertical grid
        for i in range(-100,100,1):
            soy = np.stack([np.zeros(_soyrange.shape) + (i/10.0), _soyrange], 1)
            soy = _A1(soy)
            axs[1, 2].plot(soy[:,0], soy[:,1], color = 'gray')
            soy = np.stack([_soyrange, np.zeros(_soyrange.shape) + (i/10.0)], 1)
            soy = _A1(soy)
            axs[1, 2].plot(soy[:,0], soy[:,1], color = 'gray')
        axs[1, 2].plot(func1[:,0], func1[:,1], color = 'blue')     # transform of the entire function 1
        axs[1, 2].plot(func2[:,0], func2[:,1], color = 'orange')   # transform of the entire function 2
        axs[1, 2].scatter(l1[:50,0], l1[:50,1], color='blue')      # transform of the points 1
        axs[1, 2].scatter(l1[50:,0], l1[50:,1], color='orange')    # transform of the points 2
        axs[1, 2].plot(_xrange, (-b1.value[0,0] -w1.value[0,0] * _xrange) / w1.value[1,0]) # logistic reg line
        axs[1, 2].set_title('hidden transformation')
        axs[1, 2].set_xlim(-0.1,1.1)
        axs[1, 2].set_ylim(-0.1,1.1)

        for ax in axs.flat:
            ax.label_outer()

        #plt.show()

        counter = str(_c).rjust(5, '0')
        plt.savefig(fig_dir + f'\\{counter}.png', bbox_inches='tight')
        plt.close('all')
    _c += 1 












# plot X: transformation as interpolation between parts - (results in Colah blog visualization)
# 1. first, start with the original data which is x coordinates, sine coordinates
# 2. next, apply linear transformation
# 3. next, add bias (which offsets both x, y due to equational form)
# 4. next, apply sigmoid
# repeat per layer 
# NOW: at each step above we will have a matrix of shape (25, 1). 
# to get a smooth transition that describes the transformation, we need to interpolate 
# between them, say 10 steps from graph 1 to graph 2, 10 steps from 2 to 3, etc...

np.array([[1.2,-2.0],[1.5,1.8]])
part1 = lambda x: x 
part2 = lambda x: x @ np.array([[1.2,-2.0],[1.5,1.8]]) #w0.value 
part3 = lambda x: x @ np.array([[1.2,-2.0],[1.5,1.8]]) #w0.value #+ b0.value
part4 = lambda x: _sigmoid(x @ np.array([[1.2,-2.0],[1.5,1.8]]))#w0.value) #  + b0.value # because my lines dont go that far :P

fig_dir = r'C:\Users\pwnag\Desktop\sup\deep_larn\my_plots'

def f(_steps, _indices, gryd, _x, fun1, fun2, _prefix):
    #x1 = fun1(_x)
    #x2 = fun2(_x)
    #_delta = (x2 - x1) / _steps 
    for i in _indices:
        for g in gryd:
            soy1 = fun1(g)
            _soy_delta = (fun2(g) - soy1) * (i / _steps)
            soy = soy1 + _soy_delta
            plt.plot(soy[:,0], soy[:,1], color = 'gray', linewidth=0.5)
        #res = x1 + _delta * i
        #plt.plot(res[:50,0], res[:50,1])
        #plt.plot(res[50:,0], res[50:,1])
        #plt.xticks([])  
        #plt.yticks([]) 
        plt.ylim(-1.0,1.0)
        plt.xlim(-1.0,1.0)
        #plt.show()
        counter = str(i).rjust(5, '0')
        plt.savefig(fig_dir + f'\\{_prefix}_{counter}.png', bbox_inches='tight')
        plt.close('all')



_stepsu = 100
f(_stepsu, np.arange(_stepsu + 1), gryd, X, part1, part2, '1')
#f(100, np.arange(101), gryd, X, part2, part3, '2')
f(_stepsu, np.arange(_stepsu + 1), gryd, X, part3, part4, '3')


# logarithmic interpolation - weird, maybe it won't be necessary if i use normalized data & grid?
# i think my data had scale -10,10 and grid step size was 1 so i didn't see much of the curvy parts
np.power(1.05925372518, 200)
eles = []
for i in range(200):
    first_element = 100_000 - np.power(1.05925372518, i) 
    eles.append(int(first_element))
eles = sorted(list(set(eles)))
f(100_000, eles, gryd, X, part3, part4, '3')







images = []
_c = 0
for file_name in os.listdir(fig_dir):
    if _c % 1 == 0:
        file_path = os.path.join(fig_dir, file_name)
        images.append(imageio.imread(file_path))
    _c += 1
imageio.mimsave(fig_dir + '\\movie7.gif', images, duration = 0.05)





# so I plotted the transformation from x1,x2 to h1,h2 coordinates for the sigmoid 
# activation. we first have a linear transform, and then two sigmoids appear like
# on a 3D graph, and they squish everything to 0,1 except whatever stuff is in
# their middle. 
# it looks like this: plot 2 sigmoids on a 3D graph. their middle section will be 
# stretched to the entire graph, everything else will collapse into the margins.

# then I plotted the tanh graph which looks completely different, although it has the 
# same effect. the bending looks a lot smoother, whereas with sigmoid it looked like
# an origami fold. 

# ok now im a bit confused. doing the same graph with sigmoid made it look similar. 
# current hypothesis: i used a grid with step size 1 before, here it is 0.1 
# so that one just looks more linear. i need to do a full analysis on a good 
# normalized data set, like the spiral. 


# TODO: plot two 3D animations of the hidden neurons' sigmoids as functions of x1, x2. 
# then when you look at them from the side you will see that they become h1, h2 space. 


# TODO: study optimizers: we see some of the parameters oscillate in a seizure-inducing way.
# maybe there can be benefit to using Adam which approximates the hessian so as not to go 
# up and down this canyon. 


# TODO: what keeps the network from oscillating forever? it oscillates but each step is a bit
# different so it never has a cycle in terms of back prop params. 
