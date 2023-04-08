

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


# source domain
n = 1000
mu = [0, 0]
cov = [[20, -3], [-3, 4]]
_X_source = np.random.multivariate_normal(mu, cov, n)

# SVD of source domain
_U, _S, _Vt = np.linalg.svd(_X_source, full_matrices=False)
_S = _S.reshape([2,1]) # so we are prudent with vector dimensions

# Assign labels based on the position of data points relative to the first principal component
first_pc = _Vt[0, :]
_Y_source = np.where(_X_source @ first_pc > 0, 1, -1)
_Y_source = _Y_source.reshape([n, 1])

# target distribution: rotate source domain by 45 degrees
_deg = (np.random.choice(range(1,12), 1) * 15)[0]
theta = np.radians(_deg)
c, s = np.cos(theta), np.sin(theta) # Create a rotation matrix for 45 degrees
rotation_matrix = np.array([[c, -s], [s, c]])
_X_target = (rotation_matrix @ _X_source.T).T

# SVD of target domain
_U_tar, _S_tar, _Vt_tar = np.linalg.svd(_X_target, full_matrices=False)
_S_tar = _S_tar.reshape([2,1])

if True:
        
    # check label vector mostly in span of first left singular vector
    k = 1
    _top_k_vecs = _U[:, :k]
    _Y_norm = _Y_source / np.linalg.norm(_Y_source)
    print(np.abs(_top_k_vecs.T @ _Y_norm))

    # check that SVD gives square root of the eigenvalues of covariance matrix
    np.linalg.eig(_X_source.T @ _X_source)[0].reshape([2,1])
    _S*_S

# convert relevant data to torch tensors
X = torch.tensor(_X_source, dtype=torch.float32)
X2 = torch.tensor(_X_target, dtype=torch.float32)
Y = torch.tensor(_Y_source, dtype=torch.float32)
S = torch.tensor(_S, dtype = torch.float32)
Vt = torch.tensor(_Vt, dtype = torch.float32)
S_tar = torch.tensor(_S_tar, dtype = torch.float32)
Vt_tar = torch.tensor(_Vt_tar, dtype = torch.float32)


def custom_loss(y_pred, y_true, S, Vt, S_tar, Vt_tar, W):
    k = 1
    label_align_s = torch.sum(torch.square((S * (Vt @ W))[k:,:]))
    label_align_t = torch.sum(torch.square((S_tar * (Vt_tar @ W))[k:,:]))
    return torch.sum(torch.square(y_pred - y_true)) - label_align_s + 1000.0 * label_align_t

# Define a logistic regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1, bias=False)
        self.linear.weight.data.fill_(0) # paper requires init to zero
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Create a model and train it
model = LogisticRegressionModel()
optimizer = optim.Adam(model.parameters(), lr = 0.1)

losses = []
for epoch in range(100):
    # Forward pass
    predictions = model(X)
    W = model.linear.weight.reshape([2,1])
    loss = custom_loss(predictions, Y, S, Vt, S_tar, Vt_tar, W)
    # loss = loss_function(predictions, Y, weight)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}], Loss: {loss.item():.4f}')

if True:
        
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x1_values = np.linspace(x1_min, x1_max, 100)
    W = model.linear.weight.detach().numpy()[0]
    x2_values = -(W[0] * x1_values) / W[1]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
    axes[0].plot(losses)
    axes[1].scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis')
    axes[1].plot(x1_values, x2_values, color='red')
    axes[1].axis('equal')
    axes[1].set_ylim(x2_min, x2_max)

    axes[2].scatter(X2[:, 0], X2[:, 1], c=Y, cmap='viridis')
    axes[2].plot(x1_values, x2_values, color='red')
    axes[2].axis('equal')
    axes[2].set_ylim(x2_min, x2_max)

    plt.tight_layout()
    plt.show()



#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################






fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
# First graph: Losses
axes[0].plot(losses)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss vs. Iteration')
# Second graph: Data points and decision boundary
#axes[1].scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis')
axes[1].scatter(X2[:, 0], X2[:, 1], c=Y, cmap='viridis')
# Plot the separating line
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x1_values = np.linspace(x1_min, x1_max, 100)
#W, b = model.linear.weight.detach().numpy()[0], model.linear.bias.item()
#x2_values = -(W[0] * x1_values + b) / W[1]
W = model.linear.weight.detach().numpy()[0]
x2_values = -(W[0] * x1_values) / W[1]
axes[1].plot(x1_values, x2_values, color='red')
axes[1].set_xlim(x1_min, x1_max)
axes[1].set_ylim(x2_min, x2_max)
axes[1].set_xlabel('X1')
axes[1].set_ylabel('X2')
axes[1].set_title('Logistic Regression with PyTorch')
plt.tight_layout()
plt.show()





# Print the learned parameters
print("Learned parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")


# Plot the source and target domain data
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(_X_source[:, 0], _X_source[:, 1], c=_Y_source, cmap='viridis', alpha=0.7)
axes[0].quiver(*mu, *first_pc, color='black', scale=5)
axes[0].set_title("Source Domain")
axes[0].axis('equal')

axes[1].scatter(_X_target[:, 0], _X_target[:, 1], c=_Y_source, cmap='viridis', alpha=0.7)
axes[1].quiver(*mu, *(first_pc @ rotation_matrix), color='black', scale=5)
axes[1].set_title("Target Domain")
axes[1].axis('equal')

plt.show()



































