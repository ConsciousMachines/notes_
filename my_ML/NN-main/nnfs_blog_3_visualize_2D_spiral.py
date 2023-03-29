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


'''
def forward_pass(_x, _act = 'tanh'):
    _input.value = _x 
    for i in _g[:-2]: i.forward()
    return [i for i in _g if (i.typ == _act or i.typ == 'matmul')]
Z1, A1, Z2, A2, Z3 = forward_pass(X)
'''


''' first working example params:
theta = np.sqrt(np.random.rand(100))*1.4*np.pi
eta           = 0.01
net_structure = [2, 2, 2, 1]
act_fns       = [None, Tanh, Tanh, Sigmoid]
'''
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


# https://scicomp.stackexchange.com/questions/7030/plotting-a-2d-animated-data-surface-on-matplotlib 
# ^ ^ ^ 3D ANIMATION


# 2D example - spiral
# generate spiral data 
# https://gist.github.com/45deg/e731d9e7f478de134def5668324c44c5
np.random.seed(0)
theta = np.sqrt(np.random.rand(100))*1.4*np.pi # np.linspace(0,1.2*np.pi,100)
r_a = (2*theta + np.pi) * 0.1
data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
r_b = (-2*theta - np.pi) * 0.1
data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
plt.scatter(data_a[:,0],data_a[:,1])
plt.scatter(data_b[:,0],data_b[:,1])
plt.show()
labels = np.append(np.zeros(100), np.ones(100), 0).reshape([200,1])
X = np.append(data_a, data_b, 0)

# ok critical observation with this dataset. [eta = 0.01 for all these]
# 1. at first the data set was generated so that most points were in the center. net couldn't learn.
# 2. once i changed the formulation to the current, it immediately performed better. 
#    theta = np.sqrt(np.random.rand(100))*1.3*np.pi # np.linspace(0,1.2*np.pi,100)
# 3. with the 4 hidden layers (2,2,2,2) it couldn't learn with seed 4. with seed 0 it learned in 1565 steps. 
# 4. similarly i just changed it to 3 hidden layers (2,2,2) and it took 1140 steps.
# 5. now 2 hidden layers (2,2) took 1308 steps. 



# set up network
eta           = 0.01
net_structure = [2, 2, 2, 1]
act_fns       = [None, Tanh, Tanh, Sigmoid]
ws, bs        = [None], [None] 
np.random.seed(422)
for i, j in zip(net_structure[:-1], net_structure[1:]):
    ws.append(Variable(np.random.randn(i, j)*0.1))
    bs.append(Variable(np.random.randn(1, j)*0.1))
_input        = Variable(X)
_out          = _input
for i in range(1, len(net_structure)):
    _out      = act_fns[i](Add(MatMul(_out, ws[i]), bs[i]))
loss          = CrossEntropy(_out, Variable(labels))
_g            = create_graph(loss, [])
for _e in range(100_000):
    loss.grad = 1.0
    for i in _g: i.forward() 
    for i in reversed(_g): i.backward()
    for i in ws[1:]: i.value -= eta * i.grad
    for i in bs[1:]: i.value -= eta * i.grad
    for i in _g: i.grad = 0
    # test
    pred = _out.value
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    accuracy = np.sum(np.abs(pred - labels))
    if accuracy == 0: 
        print('found a good point')
        break



_sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))     
def _tanh(_x):
    a = np.exp(_x)
    b = np.exp(-_x)
    return (a - b)/(a + b)
_ID = lambda x: x 
_Z1 = lambda x: x @ ws[1].value + bs[1].value             
_A1 = lambda x: _tanh(_Z1(x))     
_Z2 = lambda x: _A1(x) @ ws[2].value + bs[2].value
_A2 = lambda x: _tanh(_Z2(x))

fig_dir = r'C:\Users\pwnag\Desktop\sup\deep_larn\my_plots'

def f(_steps, _indices, gryd, _x, fun1, fun2, _prefix):
    x1 = fun1(_x)
    x2 = fun2(_x)
    _delta = (x2 - x1) / _steps 
    for i in _indices:
        for g in gryd:
            soy1 = fun1(g)
            _soy_delta = (fun2(g) - soy1) * (i / _steps)
            soy = soy1 + _soy_delta
            plt.plot(soy[:,0], soy[:,1], color = 'gray', linewidth=0.5)
        res = x1 + _delta * i
        plt.scatter(res[:100,0], res[:100,1])
        plt.scatter(res[100:,0], res[100:,1])
        plt.xticks([])  
        plt.yticks([]) 
        #plt.ylim(-1.0,1.0)
        #plt.xlim(-1.0,1.0)
        #plt.show()
        counter = str(i).rjust(5, '0')
        plt.savefig(fig_dir + f'\\{_prefix}_{counter}.png', bbox_inches='tight')
        plt.close('all')



_stepsu = 20
f(_stepsu, np.arange(_stepsu + 1), gryd, X, _ID, _Z1, '1')
f(_stepsu, np.arange(_stepsu + 1), gryd, X, _Z1, _A1, '2')
f(_stepsu, np.arange(_stepsu + 1), gryd, X, _A1, _Z2, '3')
f(_stepsu, np.arange(_stepsu + 1), gryd, X, _Z2, _A2, '4')



images = []
_c = 0
for file_name in os.listdir(fig_dir):
    if _c % 1 == 0:
        file_path = os.path.join(fig_dir, file_name)
        images.append(imageio.imread(file_path))
    _c += 1
for i in range(10): 
    images.append(images[-1]) # so it doesnt restart too fast
imageio.mimsave(fig_dir + '\\spiral_2.gif', images, duration = 0.05)



