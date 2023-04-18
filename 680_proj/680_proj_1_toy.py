

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, TensorDataset


def add_bias_column(x):
    return torch.cat([x, torch.ones([x.shape[0], 1])], 1)


def generate_toy_data():
        
    # source domain
    n                           = 1000
    mu                          = [0, 0]
    cov                         = [[20, -3], [-3, 4]]
    _X_source                   = np.random.multivariate_normal(mu, cov, n)
    _X_source                   = torch.tensor(_X_source, dtype = torch.float32)

    # assign labels based on the position of data points relative to the first principal component
    _, _, _Vt                   = torch.linalg.svd(_X_source, full_matrices = False)
    first_pc                    = _Vt[0, :]
    _Y_source                   = torch.where(_X_source @ first_pc > 0, 1.0, -1.0)
    _Y_source                   = _Y_source.reshape([-1, 1])

    # target distribution: rotate source domain 
    _deg                        = (np.random.choice(range(1,12), 1) * 15)[0]
    theta                       = np.radians(_deg)
    c, s                        = np.cos(theta), np.sin(theta) # create rotation matrix
    rotation_matrix             = torch.tensor([[c, -s], [s, c]], dtype = torch.float32)
    _X_target                   = (rotation_matrix @ _X_source.T).T

    return add_bias_column(_X_source), _Y_source, add_bias_column(_X_target)


def get_svd(_X_source, _X_target):

    # SVD of source domain
    U, S, Vt                        = torch.linalg.svd(_X_source, full_matrices = False)
    S                               = S.reshape([-1,1]) # make its shape (n, 1)
    
    # SVD of target domain
    U_tar, S_tar, Vt_tar            = torch.linalg.svd(_X_target, full_matrices = False)
    S_tar                           = S_tar.reshape([-1,1]) # make its shape (n, 1)

    return U, S, Vt, U_tar, S_tar, Vt_tar


def check_label_mostly_in_span_of_top_singular_vectors(_Y_source, U):
    return U.T @ (_Y_source / torch.linalg.norm(_Y_source))


def custom_loss(y_pred, y_true, S, Vt, S_tar, Vt_tar, W, k, REGULARIZE = True, lam = 10.0):
    sse               = torch.sum(torch.square(y_pred - y_true)) # sum squared error
    if REGULARIZE:
        # note: good results w lam = 10. better results w lam = 1 and excluding align_s
        label_align_s = torch.sum(torch.square((S * (Vt @ W))[k:,:]))
        label_align_t = torch.sum(torch.square((S_tar * (Vt_tar @ W))[k:,:]))
        sse           = sse - label_align_s + lam * label_align_t 
    return sse


########################### recreating the toy example ######################################
#############################################################################################
#############################################################################################
#############################################################################################


class LinearClassifier(nn.Module):
    def __init__(self, width):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(width, 1, bias = False)
        self.linear.weight.data.fill_(0) # paper requires init to zero
        
    def forward(self, x):
        return self.linear(x)
        # return torch.tanh(self.linear(x))


_X_source, _Y_source, _X_target = generate_toy_data()

U, S, Vt, U_tar, S_tar, Vt_tar = get_svd(_X_source, _X_target)

check_label_mostly_in_span_of_top_singular_vectors(_Y_source, U)

# create a model and train it
epochs          = 100
model           = LinearClassifier(_X_source.shape[1])
optimizer       = optim.Adam(model.parameters(), lr = 0.1) # SGD doesnt work?
losses          = []
for epoch in range(epochs):
    # forward pass
    predictions = model(_X_source)
    W           = model.linear.weight.reshape([-1,1])
    loss        = custom_loss(predictions, _Y_source, S, Vt, S_tar, Vt_tar, W, k=1, REGULARIZE=True)
    
    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}], Loss: {loss.item():.4f}')


# graph        
x1_min, x1_max = _X_source[:, 0].min() - 1, _X_source[:, 0].max() + 1
x2_min, x2_max = _X_source[:, 1].min() - 1, _X_source[:, 1].max() + 1
x1_values      = np.linspace(x1_min, x1_max, 100)
W              = model.linear.weight.detach().numpy()[0]
x2_values      = -(W[0] * x1_values + W[2]) / W[1]
fig, axes      = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
_ = axes[0].plot(losses)
_ = axes[0].set_title('loss')

_ = axes[1].scatter(_X_source[:, 0], _X_source[:, 1], c=_Y_source, cmap='viridis')
_ = axes[1].plot(x1_values, x2_values, color='red')
_ = axes[1].axis('equal')
_ = axes[1].set_ylim(x2_min, x2_max)
_ = axes[1].set_title('source domain')

_ = axes[2].scatter(_X_target[:, 0], _X_target[:, 1], c=_Y_source, cmap='viridis')
_ = axes[2].plot(x1_values, x2_values, color='red')
_ = axes[2].axis('equal')
_ = axes[2].set_ylim(x2_min, x2_max)
_ = axes[2].set_title('target domain')

_ = plt.tight_layout()
# _ = plt.savefig(r'/home/chad/Desktop/graph.png')
_ = plt.show()

