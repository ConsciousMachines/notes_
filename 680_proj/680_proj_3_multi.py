

import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
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


# Figure 1: project one-hot vectors on the singular vectors
if False:
        
    # normalize the one-hot vectors
    labels_n = labels / torch.linalg.vector_norm(labels, dim=0)

    # project them onto U, to get their components in each singular vector
    components = U_s.T @ labels_n

    # list of specific k values at which to cutoff/zero-out regularization (instead of a general k)
    _specific_k_values = []

    cmap = plt.cm.get_cmap('coolwarm', 10)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3.75))
    for i in range(10):

        # components for the i^th one-hot vector
        components_i = components[:,i]

        # how much of the label is actually in the span of U (norm of components, say .95)
        comp_norm_i = torch.linalg.vector_norm(components_i).item()

        # find k so that first k singular vecs span 90% of the component norm
        k = 1
        while torch.linalg.vector_norm(components_i[:k]) < 0.9*comp_norm_i:
            k = k + 1
        _specific_k_values.append(k)

        print(f'label {i} has .9 of its projection norm in the top {k} singular vectors')

        cumulative_norm = torch.sqrt(torch.cumsum(torch.square(components_i), dim=0)).cpu()
        abs_components = torch.abs(components_i).cpu()

        _ = axes[0].plot(cumulative_norm[:50], label=str(i), color=cmap(i))
        _ = axes[1].plot(abs_components[:50], label=str(i), color=cmap(i))
    _ = axes[0].set_title('cumulative norm of $U^T y_i$')
    _ = axes[1].set_title('absolute value of $U^T y_i$')
    _ = axes[0].set_ylabel('norm of $U^T y_i$')
    _ = axes[1].set_ylabel('dot product with $y_i$')
    _ = axes[0].set_xlabel('projected on top $k$ vectors')
    _ = axes[1].set_xlabel('singular vector index')
    _ = plt.tight_layout()
    _ = axes[0].legend()
    _ = axes[1].legend()
    plt.savefig('680_proj/draft/img/cumulative_norm.png')
    plt.show()

    # create a mask we can multiply by our answer to zero-out imporant vectors
    # this allows us to use a different k for each label, rather than a single k
    mask = torch.ones([785, len(digits)]).to(device)
    for _i, _k in enumerate(_specific_k_values):
        mask[:_k,_i] = 0


# optimal hard thresholding (Gavish & Donoho) -> k_target = 96
if False:
    # https://github.com/erichson/optht

    def optht(beta, sv, sigma=None):
        """Compute optimal hard threshold for singular values.
        Off-the-shelf method for determining the optimal singular value truncation
        (hard threshold) for matrix denoising.
        The method gives the optimal location both in the case of the known or
        unknown noise level.
        Parameters
        ----------
        beta : scalar or array_like
            Scalar determining the aspect ratio of a matrix, i.e., ``beta = m/n``,
            where ``m >= n``.  Instead the input matrix can be provided and the
            aspect ratio is determined automatically.
        sv : array_like
            The singular values for the given input matrix.
        sigma : real, optional
            Noise level if known.
        Returns
        -------
        k : int
            Optimal target rank.
        Notes
        -----
        Code is adapted from Matan Gavish and David Donoho, see [1]_.
        References
        ----------
        .. [1] Gavish, Matan, and David L. Donoho.
        "The optimal hard threshold for singular values is 4/sqrt(3)"
            IEEE Transactions on Information Theory 60.8 (2014): 5040-5053.
            http://arxiv.org/abs/1305.5870
        """
        # Compute aspect ratio of the input matrix
        if isinstance(beta, np.ndarray):
            m = min(beta.shape)
            n = max(beta.shape)
            beta = m / n

        # Check ``beta``
        if beta < 0 or beta > 1:
            raise ValueError('Parameter `beta` must be in (0,1].')

        if sigma is None:
            # Sigma is unknown
            # Approximate ``w(beta)``
            coef_approx = _optimal_SVHT_coef_sigma_unknown(beta)
            # Compute the optimal ``w(beta)``
            coef = (_optimal_SVHT_coef_sigma_known(beta)
                    / np.sqrt(_median_marcenko_pastur(beta)))
            # Compute cutoff
            cutoff = coef * np.median(sv)
        else:
            # Sigma is known
            # Compute optimal ``w(beta)``
            coef = _optimal_SVHT_coef_sigma_known(beta)
            # Compute cutoff
            cutoff = coef * np.sqrt(len(sv)) * sigma
        # Log cutoff and ``w(beta)``
        # Compute and return rank
        greater_than_cutoff = np.where(sv > cutoff)
        if greater_than_cutoff[0].size > 0:
            k = np.max(greater_than_cutoff) + 1
        else:
            k = 0
        return k


    def _optimal_SVHT_coef_sigma_known(beta):
        """Implement Equation (11)."""
        return np.sqrt(2 * (beta + 1) + (8 * beta)
                    / (beta + 1 + np.sqrt(beta**2 + 14 * beta + 1)))


    def _optimal_SVHT_coef_sigma_unknown(beta):
        """Implement Equation (5)."""
        return 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43


    def _mar_pas(x, topSpec, botSpec, beta):
        """Implement Marcenko-Pastur distribution."""
        if (topSpec - x) * (x - botSpec) > 0:
            return np.sqrt((topSpec - x) *
                        (x - botSpec)) / (beta * x) / (2 * np.pi)
        else:
            return 0


    def _median_marcenko_pastur(beta):
        """Compute median of Marcenko-Pastur distribution."""
        botSpec = lobnd = (1 - np.sqrt(beta))**2
        topSpec = hibnd = (1 + np.sqrt(beta))**2
        change = 1

        while change & ((hibnd - lobnd) > .001):
            change = 0
            x = np.linspace(lobnd, hibnd, 10)
            y = np.zeros_like(x)
            for i in range(len(x)):
                yi, err = integrate.quad(
                    _mar_pas,
                    a=x[i],
                    b=topSpec,
                    args=(topSpec, botSpec, beta),
                )
                y[i] = 1.0 - yi

            if np.any(y < 0.5):
                lobnd = np.max(x[y < 0.5])
                change = 1

            if np.any(y > 0.5):
                hibnd = np.min(x[y > 0.5])
                change = 1

        return (hibnd + lobnd) / 2.


    def get_k_gavish_donoho(x_t, _sigma):
        _x = x_t.cpu().numpy()
        _, _s, _ = np.linalg.svd(_x, full_matrices=False)
        _k = optht(_x, sv=_s, sigma=_sigma)
        return _k

    k = get_k_gavish_donoho(x_t, _sigma=1)
    print('Gavish Donoho k:', k)


# Figure 2: k1 vs k2 for different values of lam
if False:


    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    cmap = plt.cm.get_cmap('coolwarm', 10)

    if True:
        with open('680_proj/scores_k1_k2_lam=10000', 'rb') as f:
            scores = pickle.load(f)
        diff_k1s = sorted(set(scores[:,0]))
        for _i, diff_k1 in enumerate(diff_k1s):
            idx = scores[:,0] == diff_k1
            scores_subset = scores[idx,:]
            acc_t = scores_subset[:,4]
            _ = axs[0, 0].plot(acc_t, label=f'{int(diff_k1)}', color=cmap(_i))
        _ = axs[0, 0].set_title('$k_t$ vs acc, for different $k_s. \lambda = 10^4$')

    if True:
        with open('680_proj/scores_k1_k2_lam=100000', 'rb') as f:
            scores = pickle.load(f)
        diff_k1s = sorted(set(scores[:,0]))
        for _i, diff_k1 in enumerate(diff_k1s):
            idx = scores[:,0] == diff_k1
            scores_subset = scores[idx,:]
            acc_t = scores_subset[:,4]
            _ = axs[0, 1].plot(acc_t, label=f'{int(diff_k1)}', color=cmap(_i))
        _ = axs[0, 1].set_title('$k_t$ vs acc, for different $k_s. \lambda = 10^5$')

    if True:
        with open('680_proj/scores_k1_k2_lam=1000000', 'rb') as f:
            scores = pickle.load(f)
        diff_k1s = sorted(set(scores[:,0]))
        for _i, diff_k1 in enumerate(diff_k1s):
            idx = scores[:,0] == diff_k1
            scores_subset = scores[idx,:]
            acc_t = scores_subset[:,4]
            _ = axs[0, 2].plot(acc_t, label=f'{int(diff_k1)}', color=cmap(_i))
        _ = axs[0, 2].set_title('$k_t$ vs acc, for different $k_s. \lambda = 10^6$')

    if True:
        with open('680_proj/scores_k1_k2_lam=10000000', 'rb') as f:
            scores = pickle.load(f)
        diff_k1s = sorted(set(scores[:,0]))
        for _i, diff_k1 in enumerate(diff_k1s):
            idx = scores[:,0] == diff_k1
            scores_subset = scores[idx,:]
            acc_t = scores_subset[:,4]
            _ = axs[1,0].plot(acc_t, label=f'{diff_k1}', color=cmap(_i))
        _ = axs[1,0].set_title('$k_t$ vs acc, for different $k_s. \lambda = 10^7$')

    if True:
            
        with open('680_proj/scores_k1_k2_lam=100000000', 'rb') as f:
            scores = pickle.load(f)
        diff_k1s = sorted(set(scores[:,0]))
        for _i, diff_k1 in enumerate(diff_k1s):
            idx = scores[:,0] == diff_k1
            scores_subset = scores[idx,:]
            acc_t = scores_subset[:,4]
            _ = axs[1, 1].plot(acc_t, label=f'{diff_k1}', color=cmap(_i))
        _ = axs[1, 1].set_title('$k_t$ vs acc, for different $k_s. \lambda = 10^8$')

    if True:
        with open('680_proj/scores_k2_k2_lam=1000000000', 'rb') as f:
            scores = pickle.load(f)
        diff_k1s = sorted(set(scores[:,0]))
        for _i, diff_k1 in enumerate(diff_k1s):
            idx = scores[:,0] == diff_k1
            scores_subset = scores[idx,:]
            acc_t = scores_subset[:,4]
            _ = axs[1, 2].plot(acc_t, label=f'{diff_k1}', color=cmap(_i))
        _ = axs[1, 2].set_title('$k_t$ vs acc, for different $k_s. \lambda = 10^9$')

    tick_locs = list(range(10))
    tick_labels = list(range(5,105,10))

    for _i in range(2):
        for _j in range(3):
            _ = axs[_i, _j].set_ylim(0,.6)
            _ = axs[_i, _j].legend(title = '$k_s$', loc='upper right')
            _ = axs[_i, _j].set_xlabel('$k_t$')
            _ = axs[_i, _j].set_ylabel('target accuracy')
            _ = axs[_i, _j].set_xticks(tick_locs)
            _ = axs[_i, _j].set_xticklabels(tick_labels)

    plt.tight_layout()
    plt.savefig('680_proj/draft/img/k1_vs_k2.png')
    plt.show()


# Figure 3: data sample
if False:
        
    tr_usps, te_usps = get_data('USPS', RETURN_LOADERS = True, digits=range(10))
    tr_mnst, te_mnst = get_data('MNIST', RETURN_LOADERS = True, digits=range(10))

    img_s, label_s = next(iter(tr_mnst))
    img_s = img_s.numpy()
    img_s = img_s[:,:-1]
    img_s = img_s.reshape([-1,1,28,28])

    img_t, label_t = next(iter(tr_usps))
    img_t = img_t.numpy()
    img_t = img_t[:,:-1]
    img_t = img_t.reshape([-1,1,28,28])

    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    for i, ax in enumerate(axes.flat):
        if i < (num_images // 2):
            img = img_s[i].squeeze()
            _ = ax.imshow(img, cmap='viridis')
            _ = ax.set_title(f"source: {int(label_s[i].item())}", weight='bold')
        else:
            img = img_t[i].squeeze()
            _ = ax.imshow(img, cmap='viridis')
            _ = ax.set_title(f"target: {int(label_t[i].item())}", weight='bold')
        _ = ax.axis('off')
    plt.tight_layout()
    plt.savefig('680_proj/draft/img/data_sample.png')
    plt.show()








# METHOD = 'LINEAR'          # simple linear regression
METHOD = 'REGULARIZE'      # use regularizer from paper
# METHOD = 'GENERAL'         # my generalized regularizer


k1 = None
k2s = list(range(5,105,10))
lams = [10**i for i in range(10)]

iii                            = 0
scores                         = np.zeros([len(k2s)*len(lams),5])

for lam in lams:
    for k2 in k2s:

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

            # class Classifier(nn.Module): # this nn gets .57 with no reg, .10 with reg
            #     def __init__(self, in_dim, out_dim):
            #         super(Classifier, self).__init__()
            #         self.lin1   = nn.Linear(in_dim, 400, bias = False)
            #         self.drop   = nn.Dropout(p=0.2) 
            #         self.lin2   = nn.Linear(400, out_dim)
                    
            #     def forward(self, x):
            #         x = torch.relu(self.lin1(x))
            #         x = self.drop(x)
            #         x = torch.tanh(self.lin2(x))
            #         return x

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

                # MASK (different k)
                # label_align_s     = torch.sum(torch.square((S_s * (Vt_s @ W))*mask))
                # label_align_t     = torch.sum(torch.square((S_t * (Vt_t @ W))*mask))

                # label_align_s     = torch.sum(torch.square((S_s * (Vt_s @ W))[k1:,:]))
                label_align_t     = torch.sum(torch.square((S_t * (Vt_t @ W))[k2:,:]))
                # loss              = sse - label_align_s + lam * label_align_t 
                loss              = sse + lam * label_align_t 
            elif METHOD == 'GENERAL':

                pred_t            = model(x_t)
                # my_idea_s         = torch.sum(torch.square(U_s.T[k1:,:] @ pred_s))
                my_idea_t         = torch.sum(torch.square(U_t.T[k2:,:] @ pred_t))
                # loss              = sse - my_idea_s + lam * my_idea_t
                loss              = sse + lam * my_idea_t

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
        # scores[iii,:]             = [k1, k2, lam, acc_s, acc_t]
        # iii                       = iii + 1

        # save in case computer crashes
        # with open('680_proj/scores', 'wb') as f:
            # pickle.dump(scores, f)

        # sleep(60) # cool the gpu

        print(f'accuracies: {acc_s:.2f} / {acc_t:.2f}    k1={k1}, k2={k2}, lam={int(lam)}'.replace('0.','.'))
