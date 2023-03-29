# Ok I thought I was smart with the einsum thing but the einsum mnist takes
# 35 seconds per epoch, and this autodiff system takes 7. wowzers. 
# at least the math is the same (i didnt divide by batch size here)

import numpy as np
import matplotlib.pyplot as plt
import pickle, gzip, time


def p(op, depth = 0): # print the comp graph
    line = '\t' * depth + repr(op)
    print(line)
    if op.typ != '_input':
        for v in op.ins:
            p(v, depth + 1)


def create_graph(op): # topological sort (requires _g = [])
    if op.typ == '_input':
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




# load data
def one_hot_encode(j):
    e = np.zeros((10))
    e[j] = 1.0
    return e
f = gzip.open(r'C:\Users\pwnag\Desktop\sup\nielsen\mnist.pkl.gz', 'rb')
_tr, _va, _te = pickle.load(f, encoding = "latin1")
f.close()
tr_x = [np.reshape(x, (784, 1)) for x in _tr[0]] # reshape x's
va_x = [np.reshape(x, (784, 1)) for x in _va[0]]
te_x = [np.reshape(x, (784, 1)) for x in _te[0]]
tr_y = np.array([one_hot_encode(y) for y in _tr[1]]) # one-hot-encode the y's
tr_data = list(zip(tr_x, tr_y))
va_data = list(zip(va_x, _va[1])) # list of tuples of (x,y)
te_data = list(zip(te_x, _te[1]))


# params
batch_size = 10
eta = 0.1
# layer               0 1 2 3
hidden_structure = [784,30,10]
np.random.seed(1)
B = [None] + [Variable(np.random.randn(y, 1)) for y in hidden_structure[1:]]
W = [None] + [Variable(np.random.randn(y, x)) for x, y in zip(hidden_structure[:-1], hidden_structure[1:])]

def feedforward(a): # faster version for test 
    for _i in range(1, len(B)):
        a = 1.0 / (1.0 + np.exp(-(W[_i].value @ a + B[_i].value)))
    return a

# forward pass 
_input = Variable()
out = _input
for j in range(1, len(hidden_structure)):
    out = Sigmoid(Add(MatMul(W[j], out), B[j]))
_Y = Variable()
loss = MSE(out, _Y)

p(loss)

_g = []
create_graph(loss)
_g

for _e in range(30): # for each epoch

    s = time.time()

    __indices = np.random.permutation(50_000).reshape(50_000 // batch_size, batch_size)

    for _poop in range(50_000 // batch_size):

        # create batch
        __idx = __indices[_poop]
        batch_x = _tr[0][__idx,:].T
        batch_y = tr_y[__idx,:].T


        # forward pass 
        _input.value = batch_x
        _Y.value = batch_y
        for i in _g: i.forward()             

        # backward pass 
        loss.grad = 1
        for i in reversed(_g): i.backward()

        # update weights 
        for j in range(1, len(hidden_structure)):
            B[j].value -= eta * B[j].grad 
            W[j].value -= eta * W[j].grad 

        # reset grad after backward pass 
        for i in _g: i.grad = 0

    # test on te_data
    __test_results = [(np.argmax(feedforward(x)), y) for (x, y) in te_data]
    __evaluate = sum(int(x == y) for (x, y) in __test_results)
    print(f"Epoch {_e} : {__evaluate} / {len(te_data)}\ttime: {time.time() - s}")


