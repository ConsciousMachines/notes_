

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


def get_svd(_X_source, _X_target):

    # SVD of source domain
    U, S, Vt                        = torch.linalg.svd(_X_source, full_matrices = False)
    S                               = S.reshape([-1,1]) # make its shape (n, 1)
    
    # SVD of target domain
    U_tar, S_tar, Vt_tar            = torch.linalg.svd(_X_target, full_matrices = False)
    S_tar                           = S_tar.reshape([-1,1]) # make its shape (n, 1)

    return U, S, Vt, U_tar, S_tar, Vt_tar


#################### recreating M N I S T - U S P S #########################################
#############################################################################################
#############################################################################################
#############################################################################################


def get_data(WHICH_DATA, digits = (0, 1), batch_size = 64, RETURN_LOADERS = False):

    digits = torch.Tensor(digits)

    usps_dir       = r'/home/chad/Desktop/_backups/notes/ignore/data/USPS/'
    torch_data_dir = r'/home/chad/Desktop/_backups/notes/ignore/data'

    if WHICH_DATA == 'USPS':

        # load USPS data as per https://www.kaggle.com/datasets/bistaumanga/usps-dataset
        with h5py.File(os.path.join(usps_dir, 'usps.h5') , 'r') as hf:
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
        t = torchvision.transforms.Resize([28,28])
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
    ytr       = torch.Tensor(ytr).long()
    yte       = torch.Tensor(yte).long()

    # standardize train & test data (this converts it to float)
    xtr_mean  = torch.mean(xtr)
    xtr_std   = torch.std(xtr)
    xtr       = (xtr - xtr_mean) / xtr_std
    xte_mean  = torch.mean(xte)
    xte_std   = torch.std(xte)
    xte       = (xte - xte_mean) / xte_std

    # filter data to only include the chosen digits
    tr_idx    = torch.isin(ytr, digits)
    te_idx    = torch.isin(yte, digits)
    xtr, ytr  = xtr[tr_idx], ytr[tr_idx]
    xte, yte  = xte[te_idx], yte[te_idx]

    if RETURN_LOADERS:
        # convert to TensorDataset
        tr_ds     = TensorDataset(xtr, ytr)
        te_ds     = TensorDataset(xte, yte)

        # convert to DataLoader
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

        return tr_loader, te_loader
    
    else:
        return xtr, ytr, xte, yte
        

# look at MNIST vs USPS data distribution
if False:
    pass 

    # def plot_images(data_loader, num_images=16):
    #     images, labels = next(iter(data_loader))
    #     images = images.numpy()
    #     num_rows = int(np.sqrt(num_images))
    #     num_cols = int(np.ceil(num_images / num_rows))
        
    #     fig, axes = plt.subplots(num_rows, num_cols, figsize=(1.5 * num_cols, 2 * num_rows))
    #     for i, ax in enumerate(axes.flat):
    #         if i < num_images:
    #             img = images[i].squeeze()
    #             ax.imshow(img, cmap='viridis')
    #             ax.set_title(f"Label: {labels[i].item()}")
    #         ax.axis('off')

    #     plt.tight_layout()
    #     plt.savefig(r'/home/chad/Desktop/graph.png')
    #     plt.show()

    # tr_usps, te_usps = get_data('USPS', RETURN_LOADERS = True)
    # tr_mnst, te_mnst = get_data('MNIST', RETURN_LOADERS = True)
    # plot_images(tr_usps)
    # plot_images(tr_mnst)
    # for i, (images, labels) in enumerate(tr_usps):
    #     print(images.shape)
    #     print(labels.shape)
    #     break


def get_k(comp):
    k = 1
    # using the 90% metric gives too many vectors. not sure if paper did this for real. so im capping at 10
    while (torch.linalg.vector_norm(comp[:k]) < 0.9):
        if k > 10:
            break
        k = k + 1
    return k


def custom_loss(y_pred, y_true, S, Vt, S_tar, Vt_tar, W, k, REGULARIZE, lam):
    sse                   = torch.sum(torch.square(y_pred - y_true)) # sum squared error
        
    if REGULARIZE:

        # we pass in k as the first k vectors
        if type(k) == int:
        
            label_align_s = torch.sum(torch.square((S * (Vt @ W))[k:,:]))
            label_align_t = torch.sum(torch.square((S_tar * (Vt_tar @ W))[k:,:]))

        # we pass k as a list of indices to select 
        elif type(k) == torch.Tensor:
            label_align_s = torch.sum(torch.square((S * (Vt @ W))[k,:]))
            label_align_t = torch.sum(torch.square((S_tar * (Vt_tar @ W))[k,:]))

        else:
            raise Exception('wrong type for k')

        sse               = sse - label_align_s + lam * label_align_t 
    return sse


class LinearClassifier(nn.Module):
    def __init__(self, width):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(width, 1, bias = False)
        self.linear.weight.data.fill_(0) # paper requires init to zero
        
    def forward(self, x):
        return self.linear(x)


def prepare_data(digits):
    # prepare the data
    xtr, ytr, xte, yte                = get_data('MNIST', digits)
    _X_source                         = add_bias_column(xtr.reshape([-1, 784]))
    _X_target                         = add_bias_column(xte.reshape([-1, 784]))
    _Y_source                         = ytr.reshape([-1, 1])
    # check dims, otherwise SVD will not have 784 dims
    if (_X_source.shape[0] < _X_source.shape[1]) or (_X_target.shape[0] < _X_target.shape[1]):
        print('prepare_data: not enough data points')
        return None
    # convert labels to 1/-1 for the label align property, or 0/1 for logistic
    _Y_source[_Y_source == digits[0]] = -1
    _Y_source[_Y_source == digits[1]] = 1
    _Y_source                         = _Y_source.float()
    return _X_source, _Y_source, _X_target


def evaluate_on_target_domain(digits, model):
    # evaluate on target domain
    _xte1, _yte1, _xte2, _yte2   = get_data('USPS', digits)
    xte2                         = torch.cat([_xte1, _xte2], 0)
    yte2                         = torch.cat([_yte1, _yte2], 0)
    xte2                         = add_bias_column(xte2.reshape([-1, 784]))
    _labels                      = torch.zeros([yte2.shape[0],1]) -1
    _pred                        = torch.zeros([xte2.shape[0],1]) -1
    _labels[yte2 == digits[1],:] = 1
    _pred[model(xte2) > 0.5]     = 1
    _accuracy                    = torch.sum(_pred == _labels) / xte2.shape[0]
    return _accuracy
    

def run_experiment(REGULARIZE):

    scores = torch.zeros([90,3])
    idx = 0
    for digit_1 in range(10):
        for digit_2 in [i for i in range(10) if i != digit_1]:
            digits = [digit_1, digit_2]
            print(digits)

            _X_source, _Y_source, _X_target    = prepare_data(digits)


            # choose specific k
            if REGULARIZE:  
                U, S, Vt, U_tar, S_tar, Vt_tar = get_svd(_X_source, _X_target)
                components                     = U.T @ (_Y_source / torch.norm(_Y_source))
                # specific_k = 3
                
                #specific_k                     = get_k(components)
                
                specific_k                     = torch.where(torch.abs(components) > 0.4)[0]
                specific_k                     = int(torch.max(specific_k) + 1)
                print(type(specific_k), specific_k)
            else:
                U, S, Vt, U_tar, S_tar, Vt_tar = None, None, None, None, None, None
                specific_k                     = 0

            print('k = ', specific_k)

            # create a model and train it
            epochs                             = 1000
            model                              = LinearClassifier(_X_source.shape[1])
            optimizer                          = optim.Adam(model.parameters(), lr = 0.1) # SGD doesnt work?
            for epoch in range(epochs):
                # forward pass
                predictions                    = model(_X_source)
                W                              = model.linear.weight.reshape([-1,1])
                loss                           = custom_loss(predictions, _Y_source, S, Vt, S_tar, Vt_tar, W, k=specific_k, REGULARIZE=REGULARIZE, lam = 1000.0)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            _accuracy                          = evaluate_on_target_domain(digits, model)
            scores[idx,:]                      = torch.tensor([digit_1, digit_2, _accuracy])
            idx                                = idx + 1
            if False:
                pass    
                # # evaluate logistic regression on source domain's test set 
                # xte                          = xte.reshape([-1, 784])
                # _labels                      = torch.zeros([yte.shape[0],1])
                # _labels[yte == digits[1],:]  = 1
                # _pred                        = torch.zeros([xte.shape[0],1])
                # _pred[model(xte) > 0.5]      = 1
                # _accuracy                    = torch.sum(_pred == _labels) / xte.shape[0]
                # _accuracy
    return scores


scores = run_experiment(True)


torch.mean(scores[:,2])
# linear, no regularizer:
'tensor(0.6178)'
# k = 3 as paper suggested
'tensor(0.6911)'
# k>0.1:
'tensor(0.6796)'
# k>0.2:
'tensor(0.6832)'
# k>0.3:
'tensor(0.6777)'
# k>0.4:
'tensor(0.6680)'
# k>0.4 incl 0:
'tensor(0.7017) - yay'
# k>0.1 incl 0:
'tensor(0.6731)'
# k>0.2 incl 0:
'tensor(0.6942)'
# k>0.3 incl 0:
'tensor(0.6936)'
# k as the paper prescribes, by amassing 0.9 of the norm. 

# QUESTION: why is performance better when we include the first vectors, even tho they have low projection norm?
#     meaning, index 0 has norm 0.1, 1 has 0.6, but excluding index 0 drops performance.