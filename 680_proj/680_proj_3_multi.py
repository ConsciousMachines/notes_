

import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt
from time import sleep


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, TensorDataset


def add_bias_column(x):
    return torch.cat([x, torch.ones([x.shape[0], 1])], 1)


def get_svd(x_s, x_t):

    # SVD of source domain
    U_s, S_s, Vt_s            = torch.linalg.svd(x_s, full_matrices = False)
    S_s                       = S_s.reshape([-1,1]) # make its shape (n, 1)
    
    # SVD of target domain
    U_t, S_t, Vt_t            = torch.linalg.svd(x_t, full_matrices = False)
    S_t                       = S_t.reshape([-1,1]) # make its shape (n, 1)

    return U_s, S_s, Vt_s, U_t, S_t, Vt_t


def evaluate(model, x_t, y_t):
    pred_t                    = torch.argmax(model(x_t), dim=1, keepdim=True)
    accuracy                  = torch.sum(pred_t == y_t) / x_t.shape[0]
    return accuracy


def get_data(WHICH_DATA, digits = (0, 1), batch_size = 64, RETURN_LOADERS = False):

    digits         = torch.Tensor(digits)

    usps_dir       = r'680_proj/data/USPS/usps.h5'
    torch_data_dir = r'680_proj/data'

    if WHICH_DATA == 'USPS':

        # load USPS data as per https://www.kaggle.com/datasets/bistaumanga/usps-dataset
        with h5py.File(usps_dir, 'r') as hf:
            train  = hf.get('train')
            xtr    = train.get('data')[:]
            ytr    = train.get('target')[:]
            test   = hf.get('test')
            xte    = test.get('data')[:]
            yte    = test.get('target')[:]

        # reshape to (-1, 1, 16, 16) for torch
        xtr        = torch.Tensor(xtr).reshape(-1, 1, 16, 16)
        xte        = torch.Tensor(xte).reshape(-1, 1, 16, 16)

        # resize to (28, 28)
        t = torchvision.transforms.Resize([28,28], antialias=True)
        xtr        = t(xtr)
        xte        = t(xte)

    if WHICH_DATA == 'MNIST':

        # download MNIST dataset
        tr_ds      = torchvision.datasets.MNIST(torch_data_dir, train=True,  download=True)
        te_ds      = torchvision.datasets.MNIST(torch_data_dir, train=False, download=True)

        # convert dataset to numpy 
        xtr, ytr   = tr_ds.data.numpy(), tr_ds.targets.numpy()
        xte, yte   = te_ds.data.numpy(), te_ds.targets.numpy()

        # reshape to (-1, 1, 28, 28) for torch
        xtr        = torch.Tensor(xtr).reshape(-1, 1, 28, 28)
        xte        = torch.Tensor(xte).reshape(-1, 1, 28, 28)

    # convert the labels too
    ytr            = torch.Tensor(ytr).long()
    yte            = torch.Tensor(yte).long()

    # standardize train & test data (this converts it to float)
    xtr_mean       = torch.mean(xtr)
    xtr_std        = torch.std(xtr)
    xtr            = (xtr - xtr_mean) / xtr_std
    xte_mean       = torch.mean(xte)
    xte_std        = torch.std(xte)
    xte            = (xte - xte_mean) / xte_std

    # filter data to only include the chosen digits
    tr_idx         = torch.isin(ytr, digits)
    te_idx         = torch.isin(yte, digits)
    xtr, ytr       = xtr[tr_idx], ytr[tr_idx]
    xte, yte       = xte[te_idx], yte[te_idx]

    # reshape into a matrix, convert y to float
    xtr            = add_bias_column(xtr.reshape([-1, 784]))
    xte            = add_bias_column(xte.reshape([-1, 784]))
    ytr            = ytr.reshape([-1, 1]).float()
    yte            = yte.reshape([-1, 1]).float()

    if RETURN_LOADERS:
        # convert to TensorDataset
        tr_ds      = TensorDataset(xtr, ytr)
        te_ds      = TensorDataset(xte, yte)

        # convert to DataLoader
        tr_loader  = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        te_loader  = DataLoader(te_ds, batch_size=batch_size, shuffle=True)

        return tr_loader, te_loader
    
    else:
        return xtr, ytr, xte, yte
    

##### try one-hot encoded classes, see if the one-hot vecs are in span of singular vecs #####
############################################################### answer: they are! ###########
#############################################################################################
#############################################################################################


# load data: source and target
digits = range(10)
x_s, y_s, x_s_te, y_s_te       = get_data('MNIST', digits)
x_t, y_t, _, _                 = get_data('USPS',  digits)


# make one-hot labels with -1/1 (this makes alignment better)
labels                         = torch.zeros([x_s.shape[0], len(digits)]) - 1
labels[torch.arange(x_s.shape[0]), y_s.to(torch.long).flatten()] = 1
del y_s


# send to gpu
device                         = torch.device('cuda')
x_s                            = x_s.to(device)
x_t                            = x_t.to(device)
y_t                            = y_t.to(device)
x_s_te                         = x_s_te.to(device)
y_s_te                         = y_s_te.to(device)
labels                         = labels.to(device)


# get svd
U_s, S_s, Vt_s, U_t, S_t, Vt_t = get_svd(x_s, x_t)


# METHOD = 'LINEAR'          # simple linear regression
# METHOD = 'REGULARIZE'      # use regularizer from paper
METHOD = 'GENERAL'         # my generalized regularizer

ks                             = list(range(5,100,10))
lams                           = [10**i for i in range(7)]

iii                            = 0
scores                         = np.zeros([len(ks)*len(lams),4])
for k in ks:
    for lam in lams:

        # create a model and train it

        if (METHOD == 'LINEAR') or (METHOD == 'REGULARIZE'):

            class Classifier(nn.Module):
                def __init__(self, in_dim, out_dim):
                    super(Classifier, self).__init__()
                    self.linear   = nn.Linear(in_dim, out_dim, bias = False)
                    self.linear.weight.data.fill_(0) # paper requires init to zero

                def forward(self, x):
                    return self.linear(x)

        elif METHOD == 'GENERAL':

            class Classifier(nn.Module):
                def __init__(self, in_dim, out_dim):
                    super(Classifier, self).__init__()
                    self.linear   = nn.Linear(in_dim, out_dim, bias = False)
                    
                def forward(self, x):
                    return torch.tanh(self.linear(x))

        else:
            raise Exception('unknown method')

        model                     = Classifier(x_s.shape[1], len(digits)).to(device)
        optimizer                 = optim.Adam(model.parameters())
        # losses                    = []
        for epoch in range(1000):
            # forward pass
            pred_s                = model(x_s)
            sse                   = torch.sum(torch.square(pred_s - labels)) 
            if METHOD == 'LINEAR':
                loss              = sse
            elif METHOD == 'REGULARIZE':
                W                 = model.linear.weight.T
                label_align_s     = torch.sum(torch.square((S_s * (Vt_s @ W))[k:,:]))
                label_align_t     = torch.sum(torch.square((S_t * (Vt_t @ W))[k:,:]))
                loss              = sse - label_align_s + lam * label_align_t 
            elif METHOD == 'GENERAL':
                pred_t            = model(x_t)
                my_idea_s         = torch.sum(torch.square(U_s.T[k:,:] @ pred_s))
                my_idea_t         = torch.sum(torch.square(U_t.T[k:,:] @ pred_t))
                loss              = sse - my_idea_s + lam * my_idea_t
            else:
                raise Exception('unknown method')

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # losses.append(loss.item())
            # if (epoch + 1) % 10 == 0:
                # print(f'Epoch [{epoch + 1}], Loss: {loss.item():.4f}')

        # evaluate model
        acc_s                     = evaluate(model, x_s_te, y_s_te).item()
        acc_t                     = evaluate(model, x_t, y_t).item()

        # add accuracy results to dataframe for later
        scores[iii,:]             = [k, lam, acc_s, acc_t]
        iii                       = iii + 1

        # save in case computer crashes
        with open('680_proj/scores', 'wb') as f:
            pickle.dump(scores, f)

        print(f'counter={iii}, accuracies: {acc_s:.2f} / .{acc_t:.2f}    k={k}, lam={lam}'.replace('0.','.'))

        sleep(60) # cool the gpu




with open('data', 'rb') as f:
    scores = pickle.load(scores)


# EXPERIMENTS
# - linear, no reg:     .86 / .10
# - linear, reg:        .42 / .22    k=3, lam=1000





'''
counter=1, accuracies: .90 / .23    k=5, lam=1
counter=2, accuracies: .88 / .17    k=5, lam=10
counter=3, accuracies: .72 / .14    k=5, lam=100
counter=4, accuracies: .53 / .39    k=5, lam=1000
counter=5, accuracies: .32 / .37    k=5, lam=10000
counter=6, accuracies: .18 / .36    k=5, lam=100000
counter=7, accuracies: .13 / .06    k=5, lam=1000000
counter=8, accuracies: .90 / .23    k=15, lam=1
counter=9, accuracies: .90 / .30    k=15, lam=10
counter=10, accuracies: .84 / .23    k=15, lam=100
counter=11, accuracies: .71 / .46    k=15, lam=1000
counter=12, accuracies: .62 / .46    k=15, lam=10000
counter=13, accuracies: .41 / .49    k=15, lam=100000
counter=14, accuracies: .19 / .13    k=15, lam=1000000
counter=15, accuracies: .90 / .36    k=25, lam=1
counter=16, accuracies: .90 / .35    k=25, lam=10
counter=17, accuracies: .86 / .29    k=25, lam=100
counter=18, accuracies: .76 / .39    k=25, lam=1000
counter=19, accuracies: .71 / .39    k=25, lam=10000
counter=20, accuracies: .54 / .38    k=25, lam=100000
counter=21, accuracies: .27 / .36    k=25, lam=1000000
counter=22, accuracies: .90 / .30    k=35, lam=1
counter=23, accuracies: .90 / .30    k=35, lam=10
counter=24, accuracies: .88 / .20    k=35, lam=100
counter=25, accuracies: .80 / .33    k=35, lam=1000
counter=26, accuracies: .74 / .38    k=35, lam=10000
counter=27, accuracies: .61 / .31    k=35, lam=100000
counter=28, accuracies: .31 / .24    k=35, lam=1000000
counter=29, accuracies: .91 / .29    k=45, lam=1
counter=30, accuracies: .90 / .32    k=45, lam=10
counter=31, accuracies: .89 / .24    k=45, lam=100
counter=32, accuracies: .81 / .29    k=45, lam=1000
counter=33, accuracies: .75 / .35    k=45, lam=10000
counter=34, accuracies: .61 / .24    k=45, lam=100000
counter=35, accuracies: .39 / .26    k=45, lam=1000000
counter=36, accuracies: .91 / .29    k=55, lam=1
counter=37, accuracies: .90 / .29    k=55, lam=10
counter=38, accuracies: .89 / .13    k=55, lam=100
'''









































# EXPERIMENTS: source/target. keep k=3, lam=1000
# no regularization
# - linear             .86 / .11
# - tanh               .91 / .40

# REGULARIZER (as in paper)
# - linear             .39 / .32 (improves, as paper implied)
# - tanh               .42 / .39

# GENERALIZED LOSS (the one that makes mathematical sense)
# TODO: REDO EVERYTHING 

# TODO: make graph of k in range(100) since we have a 53% max

# TODO: nn when computer cools down. needs parameter grid search again -.-
# TODO: basic CNN

# TODO: redo basic experiments for massive lam 10**8,9,...

# TODO: compare my best k against 300 by gavish donoho (because cutting off at 90% is bad)
# https://www.youtube.com/watch?v=epoHE2rex0g&list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv&index=37
# cite 2 more things: Brunton, and Gavish









# function to copy text to clipboard (linux)
import subprocess
def copy_to_clipboard(text):
    p = subprocess.Popen(['xsel', '-bi'], stdin=subprocess.PIPE)
    p.communicate(input=text.encode())


soy = r'''


a. If the genotype makes its first appearance on the 53rd subject analyzed, then the first 52 subjects do not have the genotype, and the 53rd subject has the genotype. Assuming the subjects are independent and have the same prevalence probability $\theta$, the likelihood function can be modeled using a geometric distribution. The probability mass function (PMF) of a geometric distribution is:

$P(X = k) = (1 - \theta)^{(k - 1)} \cdot \theta$

In this case, $k = 53$. So the likelihood function $L(\theta)$ is:

$L(\theta) = (1 - \theta)^{(53 - 1)} \cdot \theta$

b. If the scientists had planned to stop when they found five subjects with the genotype of interest, and they analyzed 552 subjects, we can model this using a negative binomial distribution. The PMF of a negative binomial distribution is:

$P(X = k) = C(k - 1, r - 1) \cdot \theta^r \cdot (1 - \theta)^{(k - r)}$

In this case, $r = 5$ (the number of successes or genotypes of interest), and $k = 552$ (the number of trials). So the likelihood function $L(\theta)$ is:

$L(\theta) = C(552 - 1, 5 - 1) \cdot \theta^5 \cdot (1 - \theta)^{(552 - 5)}$

c. We can plot both likelihood functions in R:

The plot will show the likelihood functions for both scenarios a and b. You will notice that the likelihood function in scenario a, where the genotype appears on the 53rd subject, is more spread out with a lower peak than the likelihood function in scenario b, where the scientists stop after finding five subjects with the genotype. This indicates that the data in scenario b provides more information about the prevalence probability $\theta$, resulting in a more concentrated likelihood function around the most likely value of $\theta$.

'''

soy = soy.replace('θ', '$\\theta$').replace('\n', '\n\n')

print(soy)
copy_to_clipboard(soy)







'''
if False: # plot the cumulative norm contained in top singular vectors
        
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    for i in range(10):

        vec = labels[:,i]
        vec = vec / torch.linalg.vector_norm(vec)
        comp = U.T @ vec

        k = 1
        while (torch.linalg.vector_norm(comp[:k]) < 0.9) and (k < _X_source.shape[1]):
            k = k + 1
        print(f'label {i} has {int(torch.linalg.vector_norm(comp[:k]) * 100)}% of its norm in the top {k} singular vecs')

        _ = axes[0].plot(torch.sqrt(torch.cumsum(torch.pow(comp,2), 0))[:50])
        _ = axes[1].plot(torch.sqrt(torch.pow(comp,2))[:50])
    _ = plt.title('cumulative norm of label projected on top singular vecs')
    _ = plt.tight_layout()
    plt.show()
'''
