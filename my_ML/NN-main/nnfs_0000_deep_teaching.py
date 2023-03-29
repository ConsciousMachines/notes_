# from deep learning 3 (japanese)
import numpy as np


class Variable:
    def __init__(self, data):
        assert isinstance(data, np.ndarray)
        self.data = data
        self.grad = None
        self.creator = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, input):
        self.input = input
        self.output = Variable(self.forward(input.data))
        self.output.set_creator(self)
        return self.output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        return gy * 2 * self.input.data






#### MY RETARDED VERSION #################################################################
#### MY RETARDED VERSION #################################################################

class Variable:
    def __init__(self, data):
        self.data = data
        self.primal = None
        self.creator = None


class Op:
    def __call__(self, in1, in2 = None):
        self.in1 = in1
        self.in2 = in2
        self.output = Variable(self.forward())
        self.output.creator = self
        return self.output
    def forward():
        raise NotImplementedError
    def backward(g):
        raise NotImplementedError

class Add(Op):
    def forward(self):
        return self.in1.data + self.in2.data
    def backward(self, g):
        return [g ,g]

class Mul(Op):
    def forward(self):
        return self.in1.data * self.in2.data
    def backward(self, g):
        return [g*self.in2.data, g*self.in1.data]

class Square(Op):
    def forward(self):
        return self.in1.data * self.in1.data
    def backward(self, g):
        return g * 2 * self.in1.data

class Exp(Op):
    def forward(self):
        return np.exp(self.in1.data)
    def backward(self, g):
        return g * np.exp(self.in1.data)

a = Variable(2)
b = Variable(3)
c = Variable(4)
d = Square(a)



# deep teaching autodiff tutorial 1/3

#### DEEP TEACHING ##########################################################
#### DEEP TEACHING ##########################################################

# I think so far the recursive bs causes copies of computation - think of a diamond graph, then the bottom node is computed twice

import numpy as np
from matplotlib import pyplot as plt

import operator
def combine_dicts(a, b, op=operator.add): # SEVNOTE: operator is add because we combine dicts across paths, so add paths
    x = (list(a.items()) + list(b.items()) +
        [(k, op(a[k], b[k])) for k in set(b) & set(a)])
    return {x[i][0]: x[i][1] for i in range(0, len(x))}

def combine_dicts2(a,b):
    c = { i: a[i] + b[i] for i in set(b) & set(a)}
    for i in set(a) - set(b): 
        c[i] = a[i]
    for i in set(b) - set(a): 
        c[i] = b[i]
    return c


class Scalar(object): # same as Node() class in the video
    """Simple Scalar object that supports autodiff."""
    
    def __init__(self, value, name=None):
        self.value = value
        if name: # we want only the partial derivatives w.r.t. scalar instances which have a name.
            self.grad = lambda g : {name : g}
        else:
            self.grad = lambda g : {}
            
    def __add__(self, o):
        assert isinstance(o, Scalar)
        # forward pass: addition is simple:
        ret = Scalar(self.value + o.value)
        
        def grad(g): # g is the backpropagated value
            x = self.grad(g * 1) # we put the `* 1` to make it more clear: d(a+b)\da  = 1 
            x = combine_dicts(x, o.grad(g * 1))
            return x
        ret.grad = grad
        
        return ret

    def __sub__(self, o):
        assert isinstance(o, Scalar)
        ret = Scalar(self.value - o.value)
        ret.grad = lambda g: combine_dicts(self.grad(g), o.grad(-g))
        return ret 

    def __mul__(self, o):
        assert isinstance(o, Scalar)
        ret = Scalar(self.value * o.value)

        ret.grad = lambda g: combine_dicts(self.grad(g * o.value), o.grad(g * self.value)) 
        return ret

    def square(self):
        ret = Scalar(self.value * self.value)

        ret.grad = lambda g: self.grad(g * 2 * self.value)
        return ret 


 # Chain basic operations to construct a graph
a = Scalar(1., 'a')
b = Scalar(2., 'b')
c = b * a 
d = c + Scalar(1.)
# Compute the derivatives of d with respect to a and b
derivatives = d.grad(1)
assert d.value == 3.
assert d.grad(1)['a']==2.0
assert d.grad(1)['b']==1.0
a = Scalar(1., 'a')
b = Scalar(3., 'b')
c = b.square() * a 
d = (c + Scalar(1.)).square()
# test forward pass 
assert d.value == 100.
# test backward pass 
assert d.grad(1)['a'] == 180
assert d.grad(1)['b'] == 120


# generate some train data
x_min = -10.
x_max = 10.
m = 10

x = np.random.uniform(x_min, x_max, m)
a = 10.
c = 5.
y_noise_sigma = 3.
y = a + c * x + np.random.randn(len(x)) * y_noise_sigma

plt.plot(x, y, "bo")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# B A T C H 
eta = -0.001
w = Scalar(4., 'w')
b = Scalar(9., 'b')
costs = []
for j in range(200):
    grads_w = []
    grads_b = []
    pre_costs = []
    for i in range(len(x)):
        xi = Scalar(x[i])
        yi = Scalar(y[i])
        cost = (yi - ((xi * w) + b)).square()
        #costs.append(cost.value)
        pre_costs.append(cost.value)
        grad = cost.grad(1.0)
        grads_w.append(grad['w'])
        grads_b.append(grad['b'])
    costs.append(np.sum(pre_costs))
    w.value += eta * np.sum(grads_w)/10
    b.value += eta * np.sum(grads_b)/10
    
plt.plot(costs)
plt.show()








# O N L I N E 
eta = -0.01
w = Scalar(0., 'w')
b = Scalar(0., 'b')
costs = []
for j in range(200):
    for i in range(len(x)):
        xi = Scalar(x[i])
        yi = Scalar(y[i])
        cost = (yi - ((xi * w) + b)).square()
        costs.append(cost.value)
        grad = cost.grad(1.0)
        w.value += eta * grad['w']
        b.value += eta * grad['b']
plt.plot(costs)
plt.show()

y21 = w.value * (-10) + b.value
y22 = w.value * 10 + b.value
plt.plot(x, y, "bo")
plt.plot([-10,10],[y21,y22])
plt.show()





