# each Op is a node. It is a boxed variable with function. 
# it has a list of inputs, nodes pointing to it called "ins"
# it has a list of nodes that it points to called "goes_to"
# it has a numeric value computed by forward 
# it also has a grad that starts at 0
# typ is its type, like add/mul/sigmoid
# name is just type with its instance number
# when initialized, it adds itself to its inputs' goes_to list

# ok wow it worked on the first try. I made this thing after reading 
# around 10 articles on computational graphs. The only code I stole was 
# the Add function from sabinasz bc it broadcasts derivatives
# main ideas:
# 1. we should be able to traverse the entire graph forward & back from anywhere
# 2. Variables are wrappers for data, point to functions, which produce new vars
#     I ended up merging them and just made variables be an 'identity' func
# 3. the dlfs book said his graph is broken because nodes can't be reused. 
#     this requires that data has a list of places it goes to, and its grad
#     is accumulated by other funcs adding to it. a 2 line fix. 
# 4. topological sort is just depth first search. 

import numpy as np 
import matplotlib.pyplot as plt


def p(op, depth = 0): # print the comp graph
    line = '\t' * depth + repr(op)
    print(line)
    if op.typ != 'input':
        for v in op.ins:
            p(v, depth + 1)


def create_graph(op): # topological sort (requires _g = [])
    if op.typ == 'input':
        if op not in _g: # so we don't have duplicates 
            _g.append(op)
    else:
        for i in op.ins:
            create_graph(i)
        if op not in _g:
            _g.append(op)


class Op:
    c = 0
    def __init__(self, ins, typ:str = ''):
        self.ins = ins 
        self.goes_to = []

        self.value = None
        self.grad = 0 # grad with respect to node's output

        self.typ = typ
        self.name = self.make_name(typ)

        for i in self.ins:
            i.goes_to.append(self)
    @classmethod
    def make_name(cls, name):
        cls.c += 1
        return f'{name}_{cls.c}'
    def forward():
        raise NotImplementedError()
    def backward():
        raise NotImplementedError()
    def __repr__(self):
        return f'{self.name}'

class Variable(Op):
    def __init__(self, v):
        super().__init__([], 'input')
        self.value = v

class Sigmoid(Op):
    def __init__(self, v1):
        super().__init__([v1], 'sigmoid')
    def forward(self):
        self.value = 1.0 / (1.0 + np.exp(-self.ins[0].value))
    def backward(self):
        self.ins[0].grad += self.grad * self.value * (1.0 - self.value)

class MatMul(Op):
    def __init__(self, v1, v2):
        super().__init__([v1, v2], 'matmul')
    def forward(self):
        self.value = self.ins[0].value @ self.ins[1].value
    def backward(self):
        self.ins[0].grad += self.grad @ self.ins[1].value.T
        self.ins[1].grad += self.ins[0].value.T @ self.grad

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
        self.ins[0].grad += self.error
        self.ins[1].grad += -self.error











# init data
X = Variable(np.array([[0,0,1,1],[0,1,0,1]]).T)
Y = Variable(np.array([[0,1,1,0]]).T)

# initialize weights
eta = 0.2
num_input = 2
num_hidden = 2
num_output = 1
np.random.seed(0)
W1 = Variable(np.random.uniform(size=(num_input, num_hidden)))
W2 = Variable(np.random.uniform(size=(num_hidden, num_output)))
B1 = Variable(np.random.uniform(size=(1,num_hidden)))
B2 = Variable(np.random.uniform(size=(1,num_output)))

H = Sigmoid(Add(MatMul(X, W1), B1))
out = Sigmoid(Add(MatMul(H, W2), B2))
loss = MSE(out, Y)

p(loss)

_g = []
create_graph(loss)
_g


losses = []
for _e in range(8000):

    # forward pass 
    for i in _g:
        if i.typ != 'input':
            i.forward()

    # calculate cost
    losses.append(loss.value)

    # backward pass 
    loss.grad = np.array([[1]])
    for i in reversed(_g):
        if i.typ != 'input':
            i.backward()

    # update
    W2.value -= eta * W2.grad
    B2.value -= eta * B2.grad
    W1.value -= eta * W1.grad
    B1.value -= eta * B1.grad

    if _e % 500 == 0:
        print("error: ", losses[_e]) 

    # reset grad after backward pass 
    for i in _g:
        i.grad = 0

plt.plot(losses)
plt.show()


