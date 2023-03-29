# https://colinraffel.com/blog/you-don-t-know-jax.html
# there are 3 differences between my code and jax:
# 1. no grad calculations needed
# 2. network and loss code are inside functions
# 3. parameters must be in a list




import jax
import jax.numpy as np
import numpy as onp


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0,1,1,0])
eta = 1.


# feedforward
def net(params, x):
    w1, b1, w2, b2 = params
    hidden = np.tanh(np.dot(w1, x) + b1)
    return sigmoid(np.dot(w2, hidden) + b2)


# Cross-entropy loss
def loss(params, x, y):
    out = net(params, x)
    cross_entropy = -y * np.log(out) - (1 - y)*np.log(1 - out)
    return cross_entropy


loss_grad = jax.grad(loss)

params = [
        onp.random.randn(3, 2),  # w1
        onp.random.randn(3),  # b1
        onp.random.randn(3),  # w2
        onp.random.randn(),  #b2
    ]
    
for i in range(1000):
    x = X[i % 4]
    y = Y[i % 4]
    # Get the gradient
    grads = loss_grad(params, x, y)  
    # Update
    params = [p - eta * g for p, g in zip(params, grads)]

    # check whether we've solved XOR
    if i % 10 == 0:
        print('Iteration {}'.format(i))
        predictions = [int(net(params, inp) > 0.5) for inp in X]
        for inp, out in zip(X, predictions):
            print(inp, '->', out)
        if (predictions == [onp.bitwise_xor(*inp) for inp in X]): 
            break








# vmap version


loss_grad = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0), out_axes=0))

params = [
        onp.random.randn(3, 2),  # w1
        onp.random.randn(3),  # b1
        onp.random.randn(3),  # w2
        onp.random.randn(),  #b2
    ]

batch_size = 40

for i in range(1000):

    # Generate a batch of X
    x = X[onp.random.choice(X.shape[0], size=batch_size)]
    y = onp.bitwise_xor(x[:, 0], x[:, 1])

    # The call to loss_grad remains the same!
    grads = loss_grad(params, x, y)

    # Note that we now need to average gradients over the batch
    params = [param - eta * np.mean(grad, axis=0)
              for param, grad in zip(params, grads)]

    # check whether we've solved XOR
    if i % 10 == 0:
        print('Iteration {}'.format(i))
        predictions = [int(net(params, inp) > 0.5) for inp in X]
        for inp, out in zip(X, predictions):
            print(inp, '->', out)
        if (predictions == [onp.bitwise_xor(*inp) for inp in X]): 
            break
