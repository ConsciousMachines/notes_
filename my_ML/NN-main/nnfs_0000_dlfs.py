import numpy as np
from numpy import ndarray
from typing import List, Tuple
from copy import deepcopy



def assert_same_shape(a1: ndarray, a2: ndarray):
    assert a1.shape == a2.shape, f'Array shape mismatch: {a1.shape} vs {a2.shape}'


class Operation(object):
    def __init__(self):
        pass

    def _output(self) -> ndarray:
        raise NotImplementedError()

    def _input_grad(self, g: ndarray) -> ndarray:
        raise NotImplementedError()

    def forward(self, input_: ndarray): # stores input, calls self._output()
        self.input_ = input_
        self.output = self._output()
        return self.output

    def backward(self, g: ndarray) -> ndarray: # calls self._input_grad()
        assert_same_shape(self.output, g)
        self.input_grad = self._input_grad(g)
        assert_same_shape(self.input_, self.input_grad)
        return self.input_grad


class ParamOperation(Operation): # an Op that also stores its parameter
    def __init__(self, param: ndarray) -> ndarray:
        super().__init__()
        self.param = param

    def backward(self, g: ndarray) -> ndarray: # Calls self._input_grad and self._param_grad.
        assert_same_shape(self.output, g)
        self.input_grad = self._input_grad(g)
        self.param_grad = self._param_grad(g)
        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)
        return self.input_grad

    def _param_grad(self, g: ndarray) -> ndarray:
        raise NotImplementedError()


class WeightMultiply(ParamOperation):
    def __init__(self, W: ndarray):
        super().__init__(W)

    def _output(self) -> ndarray:
        return np.dot(self.input_, self.param)

    def _input_grad(self, g: ndarray) -> ndarray:
        return np.dot(g, np.transpose(self.param, (1, 0)))

    def _param_grad(self, g: ndarray)  -> ndarray: 
        return np.dot(np.transpose(self.input_, (1, 0)), g)


class BiasAdd(ParamOperation):
    def __init__(self, B: ndarray):
        assert B.shape[0] == 1
        super().__init__(B)

    def _output(self) -> ndarray:
        return self.input_ + self.param

    def _input_grad(self, g: ndarray) -> ndarray:
        return g #np.ones_like(self.input_) * g

    def _param_grad(self, g: ndarray) -> ndarray:
        #param_grad = np.ones_like(self.param) * g
        #return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1]) # sum over batch
        return np.sum(g, 0).reshape(1, g.shape[1])


class Sigmoid(Operation):
    def __init__(self) -> None:
        super().__init__()

    def _output(self) -> ndarray:
        return 1.0/(1.0+np.exp(-1.0 * self.input_))

    def _input_grad(self, g: ndarray) -> ndarray:
        return g * self.output * (1.0 - self.output)




class Linear(Operation): # Identity activation 
    def __init__(self) -> None:
        super().__init__()

    def _output(self) -> ndarray:
        return self.input_

    def _input_grad(self, g: ndarray) -> ndarray:
        return g


class Layer(object):
    def __init__(self, neurons: int):
        self.neurons = neurons
        self.first = True
        self.params: List[ndarray] = []
        self.param_grads: List[ndarray] = []
        self.operations: List[Operation] = []

    def _setup_layer(self, num_in: int) -> None:
        # NOTE: i feel like the num_in arg assumes dense layer?
        raise NotImplementedError()

    def forward(self, input_: ndarray) -> ndarray: # Passes input forward through a series of operations
        if self.first:
            self._setup_layer(input_) # initialize layers on first call 
            self.first = False

        self.input_ = input_
        for operation in self.operations:
            input_ = operation.forward(input_)
        self.output = input_
        return self.output

    def backward(self, g: ndarray) -> ndarray:
        assert_same_shape(self.output, g) # not necessary. done by last Op

        for operation in reversed(self.operations):
            g = operation.backward(g)

        self._param_grads() # extract parameter gradients from this layer's ops

        return g

    def _param_grads(self) -> ndarray: # Extracts the _param_grads from a layer's operations
        self.param_grads = [] # make it empty since it'll be new stuff 
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> ndarray: # Extracts the _params from a layer's operations
        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)




class Dense(Layer):
    def __init__(self, neurons: int, activation: Operation = Sigmoid()):
        super().__init__(neurons)
        self.activation = activation

    def _setup_layer(self, input_: ndarray) -> None: # Defines the operations of a fully connected layer.
        if self.seed:
            np.random.seed(self.seed)

        self.params.append(np.random.randn(input_.shape[1], self.neurons)) # weights
        self.params.append(np.random.randn(1, self.neurons)) # bias
        self.operations = [WeightMultiply(self.params[0]), BiasAdd(self.params[1]), self.activation]



class Loss(object):
    def __init__(self):
        pass

    def forward(self, prediction: ndarray, target: ndarray) -> float:
        assert_same_shape(prediction, target)
        self.prediction = prediction
        self.target = target
        return self._output()

    def backward(self) -> ndarray: # Computes gradient of the loss value with respect to the input to the loss function
        self.input_grad = self._input_grad()
        assert_same_shape(self.prediction, self.input_grad)
        return self.input_grad

    def _output(self) -> float:
        raise NotImplementedError()

    def _input_grad(self) -> ndarray:
        raise NotImplementedError()


class MeanSquaredError(Loss):
    def __init__(self) -> None:
        super().__init__()

    def _output(self) -> float: # Computes the per-observation squared error loss
        return np.sum(np.square(self.prediction - self.target)) / self.prediction.shape[0]

    def _input_grad(self) -> ndarray: # Computes the loss gradient with respect to the input for MSE loss
        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]


class NeuralNetwork(object):
    def __init__(self, layers: List[Layer], loss: Loss, seed: int = 1) -> None:
        self.layers = layers
        self.loss = loss
        self.seed = seed
        if seed:
            for i, layer in enumerate(self.layers):
                setattr(layer, "seed", self.seed + i)        

    def forward(self, x_batch: ndarray) -> ndarray:
        x = x_batch
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, g: ndarray) -> None:
        for layer in reversed(self.layers):
            g = layer.backward(g)

    def train_batch(self, xb: ndarray, yb: ndarray) -> float: # forward, loss, backward
        pred = self.forward(xb)
        loss = self.loss.forward(pred, yb)
        self.backward(self.loss.backward())
        return loss
    
    def params(self): # generator that yields params, for use with optimizer update
        for layer in self.layers:
            yield from layer.params

    def param_grads(self): # generator that yields param_grads, for use with optimizer update
        for layer in self.layers:
            yield from layer.param_grads    




class Optimizer:
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self) -> None:
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        super().__init__(lr)

    def step(self): # update params, requires "self.net" attribute set
        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            param -= self.lr * param_grad




class Trainer(object):
    def __init__(self, net: NeuralNetwork, optim: Optimizer) -> None: # assigns NN to optimizer
        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        setattr(self.optim, 'net', self.net)
        
    def generate_batches(self, X: ndarray, y: ndarray, size: int = 32) -> Tuple[ndarray]:
        assert X.shape[0] == y.shape[0], f'X and y length mismatch: {X.shape[0]} vs {y.shape[0]}'
        for _i in range(0, X.shape[0], size):
            X_batch, y_batch = X[_i:_i + size], y[_i:_i + size] # NOTE: the last batch will be smol if N % size =/= 0
            yield X_batch, y_batch

            
    def fit(self, X_train: ndarray, y_train: ndarray,
            X_test: ndarray, y_test: ndarray,
            epochs: int=100,
            eval_every: int=10,
            batch_size: int=32,
            seed: int = 1,
            restart: bool = True)-> None:
        '''
        Fits the neural network on the training data for a certain number of epochs.
        Every "eval_every" epochs, it evaluated the neural network on the testing data.
        '''

        np.random.seed(seed)
        if restart:
            for layer in self.net.layers:
                layer.first = True

            self.best_loss = 1e9

        for e in range(epochs):

            if (e+1) % eval_every == 0:
                
                # for early stopping
                last_model = deepcopy(self.net)

            X_train, y_train = permute_data(X_train, y_train)

            batch_generator = self.generate_batches(X_train, y_train,
                                                    batch_size)

            for _i, (X_batch, y_batch) in enumerate(batch_generator):

                self.net.train_batch(X_batch, y_batch)

                self.optim.step()

            if (e + 1) % eval_every == 0:

                test_preds = self.net.forward(X_test)
                loss = self.net.loss.forward(test_preds, y_test)

                if loss < self.best_loss:
                    print(f"Validation loss after {e+1} epochs is {loss:.3f}")
                    self.best_loss = loss
                else:
                    print(f"""Loss increased after epoch {e+1}, final loss was {self.best_loss:.3f}, using the model from epoch {e+1-eval_every}""")
                    self.net = last_model
                    # ensure self.optim is still updating self.net
                    setattr(self.optim, 'net', self.net)
                    break











# MY REMIX 

import numpy as np
from numpy import ndarray
from typing import List


def assert_same_shape(a1: ndarray, a2: ndarray):
    assert a1.shape == a2.shape, f'Array shape mismatch: {a1.shape} vs {a2.shape}'

# d/dx g(f(x)) 
# g'(f(x)) * f'(x)


class Op:
    def __init__(self) -> None:
        pass 

    def forward(self) -> ndarray:
        # this gets implemented by each specific op, and it uses its instance data to compute forward 
        raise NotImplementedError()

    def grad_x(self, g: ndarray) -> ndarray:
        # takes a gradient and passes it on 
        raise NotImplementedError()

    def forward(self, x: ndarray) -> ndarray:
        # wraps the forward() fn and saves data 
        self._x = x 
        self._f = self.forward() # this is the overloaded specific func
        return self._f

    def backward(self, g: ndarray) -> ndarray:
        # wraps grad_x, grad_w functions and saves data and checks shapes
        assert_same_shape(g, self._f)
        self._dx = self.grad_x(g)
        assert_same_shape(self._dx, self._x)
        return self._dx


class ParamOp(Op):
    def __init__(self, W: ndarray) -> None:
        super().__init__()
        self._w = W

    def grad_w(self, g: ndarray) -> ndarray:
        raise NotImplementedError()
    
    def backward(self, g: ndarray) -> ndarray:
        # overloaded from Op, because it needs to do double work for W
        assert_same_shape(g, self._f)
        self._dx = self.grad_x(g)
        self._dw = self.grad_w(g)
        assert_same_shape(self._dx, self._x)
        assert_same_shape(self._dw, self._w)
        # we extract _dw differently since they are leafs in the grad tree
        # we need grad_x to continue through but it stops at grad_w 
        return self._dx 


class WeightMul(ParamOp):
    def __init__(self, W: ndarray) -> None:
        super.__init__(W) # self._w = W
    
    def grad_w(self, g: ndarray) -> ndarray:
        return np.transpose(self._x, (1, 0)) @ g

    def grad_x(self, g: ndarray) -> ndarray:
        return g @ np.transpose(self._w, (1, 0))

    def forward(self) -> ndarray:
        return self._x @ self._w 


class BiasAdd(ParamOp):
    def __init__(self, W: ndarray) -> None: 
        assert W.shape[0] == 1
        super.__init__(W)

    def forward(self) -> ndarray:
        return self._x + self._w 

    def grad_w(self, g: ndarray) -> ndarray:
        return np.sum(g, 0).reshape(1, g.shape[1]) # TODO: is this just a flatten

    def grad_x(self, g: ndarray) -> ndarray:
        return g 


class Sigmoid(Op):
    def __init__(self) -> None: 
        super.__init__()

    def forward(self) -> ndarray:
        return 1.0 / (1.0 + np.exp(-self._x))

    def grad_x(self, g: ndarray) -> ndarray:
        return self._f * (1.0 - self._f) * g 


class Linear(Op):
    def __init__(self) -> None: 
        super.__init__()

    def forward(self) -> ndarray:
        return self._x

    def grad_x(self, g: ndarray) -> ndarray:
        return g


class Layer:
    def __init__(self, width: int) -> None:
        self.first = True # basically "uninitialized" or "first call"
        self.width = width 
        self.ops: List[Op] = []
        self.ws: List[ndarray] = []
        self.grad_ws: List[ndarray] = []

    def _setup_layer(self, num_in: int) -> None:
        raise NotImplementedError()
    
    def forward(self, x: ndarray) -> ndarray:
        if self.first:
            self._setup_layer(x) # initialize layers on first call 
            self.first = False

        self._x = x 
        for op in self.ops:
            x = op.forward(x)
        self._f = x 
        return x 

    def backward(self, g: ndarray) -> ndarray:
        for op in reversed(self.ops):
            g = op.backward(g)
        self.get_grad_ws()
        return g 

    def get_grad_ws(self) -> None:
        self.grad_ws = [] # refresh grad_ws
        for op in self.ops:
            if type(op) == ParamOp:
                self.grad_ws.append(op._dw)

    def get_ws(self) -> None:
        self.ws = [] # refresh ws
        for op in self.ops:
            if type(op) == ParamOp:
                self.ws.append(op._w)


class Dense(Layer):
    def __init__(self, width: int, act: Op = Sigmoid()):
        super().__init__(width)
        self.act = act

    def _setup_layer(self, x: ndarray) -> None: 
        if self.seed:
            np.random.seed(self.seed)
        self.ws.append(np.random.randn(x.shape[1], self.width)) # weights
        self.ws.append(np.random.randn(1, self.width)) # bias
        self.ops = [WeightMul(self.ws[0]), BiasAdd(self.ws[1]), self.act]


class Loss:
    def __init__(self):
        pass

    def forward(self, pred: ndarray, y: ndarray) -> float:
        assert_same_shape(pred, y)
        self.pred = pred 
        self.y = y 
        return self.forward() 

    def backward(self) -> ndarray:
        self.g = self.init_gradient()
        assert_same_shape(self.pred, self.g)
        return self.g

    def forward(self) -> float: 
        raise NotImplementedError()

    def init_gradient(self) -> ndarray:
        raise NotImplementedError()


class MSE(Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self) -> float: 
        return np.sum(np.square(self.pred - self.y)) / self.pred.shape[0]

    def init_gradient(self) -> ndarray:
        return 2 * (self.pred - self.y) / self.pred.shape[0]