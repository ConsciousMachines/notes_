import numpy as np 
import matplotlib.pyplot as plt


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
    def __init__(self, v):
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




# init data
X = np.array([[0,0,1,1],[0,1,0,1]]).T 
Y = np.array([[0,1,1,0]]).T 
num_inp = 2
eta = 1.

# init weights 
np.random.seed(0)
# layer                   0 1 2 3
hidden_structure = [num_inp,17,5,1]
W = [None] # so that weights of layer i will be at index i
B = [None]
for i in range(1, len(hidden_structure)):
    W.append(Variable(np.random.uniform(size=[hidden_structure[i], hidden_structure[i-1]])))
    B.append(Variable(np.random.uniform(size=[hidden_structure[i], 1])))


# forward pass 
_input = Variable(X.T)
for j in range(1, len(hidden_structure)):
    _input = Sigmoid(Add(MatMul(W[j], _input), B[j]))
loss = MSE(_input, Variable(Y.T))

p(loss)

_g = []
create_graph(loss)
_g



losses = []
for _e in range(8000):

    # forward pass 
    for i in _g: i.forward()

    # calculate cost
    losses.append(loss.value)

    # backward pass 
    loss.grad = 1
    for i in reversed(_g): i.backward()

    # update weights 
    for j in range(1, len(hidden_structure)):
        B[j].value -= eta * B[j].grad 
        W[j].value -= eta * W[j].grad 

    # reset grad after backward pass 
    for i in _g: i.grad = 0


plt.plot(losses)
plt.ylim(0, 2)
plt.show()

