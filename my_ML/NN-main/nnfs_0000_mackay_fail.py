# this tut uses MCMC which i dont get the theory behind Metropolis-Hastings

import jax
import jax.numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('ggplot')



# data
D1 = np.array([[7, 4],[5, 6],[8, 6],[9.5, 5],[9, 7]])
D2 = np.array([[2, 3],[3, 2],[3, 6],[5.5, 4.5],[5, 3]])
D_tmp = np.concatenate([D1, D2], axis=0)
D = np.concatenate((D_tmp, np.ones((10, 1))), axis=1) # 1s for intercept
t = np.concatenate((np.ones(5), np.zeros(5))) # labels

# view data
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(D1[:, 0], D1[:, 1])
ax.scatter(D2[:, 0], D2[:, 1])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set(xlim=(0, 10), ylim=(0, 10))
plt.show()

# 3d mesh grid
x_range = np.linspace(0, 10, 100)
y_range = np.linspace(0, 10, 100)
xx, yy = np.meshgrid(x_range, y_range)
Zd = np.stack([yy, xx, np.ones([100,100])], axis = 2)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def bce_loss(W):
    y = sigmoid(np.dot(D, W))
    return np.sum(- t * np.log(y) - (1 - t) * np.log(1 - y))

@jax.jit
def step(params):
    W, eta = params
    return W - eta * jax.grad(bce_loss)(W)


eta = 0.02
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
W = jax.random.normal(key, (3,)) * 0.25

fns, idx, weights = [], [], []
for _e in range(20):
    for k in range(1000):
        params = (W, eta)
        W = step(params)
        weights.append(W)
    idx.append(_e)
    fns.append(sigmoid(np.dot(Zd, W))) # Calculate and save contour lines   



fns = np.array(fns)
weights = np.array(weights)







def plot_weights(weights):
    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 6))
    # Plot weight values on x1 and x2
    ax2.set(xlim=(-1, 5), ylim=(-1, 5))
    ax2.scatter(weights[:, 0], weights[:, 1])
    ax2.set_xlabel('$w_1$')
    ax2.set_ylabel('$w_2$')
    # Plot the weights values as training continues
    ax3.plot(weights[:, 0])
    ax3.plot(weights[:, 1])
    ax3.plot(weights[:, 2])
    ax3.set_xlabel('Step')
    ax3.set_ylabel('$|w_i|$')
    plt.show()


def plot_contour(idx, fns):
    x_range = np.linspace(0, 10, 100)
    y_range = np.linspace(0, 10, 100)
    # Animation
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set(xlim=(0, 10), ylim=(0, 10))
    def animate(i):
        ax.cla()
        ax.scatter(D1[:, 0], D1[:, 1])
        ax.scatter(D2[:, 0], D2[:, 1])
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title("Step: {}".format(idx[i]))
        ax.contour(x_range, y_range, fns[i], vmin=0, vmax=1)
    anim = FuncAnimation(fig, animate, interval=200, frames=len(idx))
    plt.show()

plot_weights()
plot_contour()










def loss(W, alpha):
  return bce_loss(W) + alpha * 0.5 * (W.dot(W)).sum()

@jax.jit
def step(W, eta=0.02, alpha=0.01):
  return W - eta * jax.grad(loss)(W, alpha)


eta = 0.02
alpha = 0.01
W = jax.random.normal(key, (3,)) * 0.25
fns, idx, weights = [], [], []
for k in range(10001):
    # Calculate and save for contour lines
    if (k % 1000 == 0) or (k <= 1000 and k % 100 == 0):
        idx.append(k)
        fns.append(sigmoid(np.dot(Zd, W)))
    W = step(W, eta, alpha)
    weights.append(W)

fns = np.array(fns)
plot_contour(idx, fns)

weights = np.array(weights)
plot_weights(weights)








@jax.jit
def step(W, dW, M, k1):
    p = jax.random.normal(k1, (3,)) # noise 
    H = (p ** 2).sum() / 2 + M

    # For Hamiltonian Monte Carlo, run the following 4 lines for > 1 iterations, 
    p = p - epsilon / 2 * dW
    new_W = W + epsilon * p
    new_dW = jax.grad(loss)(new_W, alpha)
    p = p - epsilon / 2 * new_dW

    new_M = loss(new_W, alpha)
    new_H = (p ** 2).sum() / 2 + new_M

    dH = new_H - H

    return dH, new_W, new_dW, new_M, k1


eta = 0.02
alpha = 0.01
epsilon = np.sqrt(2 * eta)
W = jax.random.normal(key, (3,)) * 0.25
M = loss(W, alpha)
dW = - jax.grad(loss)(W, alpha)
k1, k2 = jax.random.split(key)
fns, idx, weights = [], [], []
for k in range(10001):
    k1, k2 = jax.random.split(k1)
    dH, new_W, new_dW, new_M, k1 = step(W, dW, M, k1)

    if (dH < 0) or (jax.random.uniform(k2) < np.exp(-dH)):
        W = new_W
        M = new_M
        dW = new_dW

    if (k > 2000 and k % 100 == 0):
        weights.append(W)
        fns.append(sigmoid(np.dot(Zd, W)))
        idx.append(k)





fns = np.array(fns)
weights = np.array(weights)

Z = fns.mean(axis=0)
plt.figure(figsize=(8, 8))
_ = plt.contour(x_range, y_range, Z, origin='lower', cmap=cm.viridis, vmin=0, vmax=1)
plt.scatter(D1[:, 0], D1[:, 1])
plt.scatter(D2[:, 0], D2[:, 1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()

Z = fns.std(axis=0)
plt.figure(figsize=(8, 8))
_ = plt.contour(x_range, y_range, Z, origin='lower', cmap=cm.viridis)
plt.scatter(D1[:, 0], D1[:, 1])
plt.scatter(D2[:, 0], D2[:, 1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()