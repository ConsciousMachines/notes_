# this is the cnn from https://roberttlange.github.io/posts/2020/03/blog-post-10/

from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import (BatchNorm, Conv, Dense, Flatten, Relu, LogSoftmax)
from jax import grad, jit, vmap, value_and_grad
from jax import random
import jax.numpy as np
import torch
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt


def plot_mnist_performance(losses, train_acc, test_acc):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(losses)
    axs[0].set_xlabel("# Batch Updates")
    axs[0].set_ylabel("Batch Loss")
    axs[0].set_title("Training Loss")
    axs[1].plot(train_acc, label="Training")
    axs[1].plot(test_acc, label="Test")
    axs[1].set_xlabel("# Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Prediction Accuracy")
    axs[1].legend()
    for i in range(2):
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
    fig.tight_layout(rect=[0, 0.03, 1, 0.925])
    plt.show()


def one_hot(x, k, dtype=np.float32):
    return np.array(x[:, None] == np.arange(k), dtype)


def loss(params, images, targets):
    return -np.sum(conv_net(params, images) * targets)


@jit
def update(params, x, y, opt_state):
    value, grads = value_and_grad(loss)(params, x, y)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, value


def accuracy(params, data_loader):
    acc_total = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        images = np.array(data)
        targets = one_hot(np.array(target), 10)
        target_class = np.argmax(targets, axis=1)
        predicted_class = np.argmax(conv_net(params, images), axis=1)
        acc_total += np.sum(predicted_class == target_class)
    return acc_total/len(data_loader.dataset)










init_fun, conv_net = stax.serial(
    Conv(32, (5, 5), (2, 2), padding="SAME"),
    BatchNorm(), 
    Relu,
    Conv(32, (5, 5), (2, 2), padding="SAME"),
    BatchNorm(), 
    Relu,
    Conv(10, (3, 3), (2, 2), padding="SAME"),
    BatchNorm(), 
    Relu,
    Conv(10, (3, 3), (2, 2), padding="SAME"), 
    Relu,
    Flatten,
    Dense(10),
    LogSoftmax)

key = random.PRNGKey(1)
batch_size = 100
step_size = 1e-3

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=batch_size, shuffle=True)

_, params = init_fun(key, (batch_size, 1, 28, 28))
opt_init, opt_update, get_params = optimizers.adam(step_size)
opt_state = opt_init(params)

params = get_params(opt_state) # init params
log_train, log_test, losses = [], [], [] # log
log_train.append(accuracy(params, train_loader)) # initial accuracy 
log_test.append(accuracy(params, test_loader))

for _e in range(10):
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        x = np.array(data)
        y = one_hot(np.array(target), 10)
        params, opt_state, loss = update(params, x, y, opt_state)
        losses.append(loss)

    log_train.append(accuracy(params, train_loader))
    log_test.append(accuracy(params, test_loader))
    print(f"Epoch {_e} | T: {time.time() - start_time} | Train A: {log_train[_e]} | Test A: {log_test[_e]}")



plot_mnist_performance(losses, log_train, log_test)
