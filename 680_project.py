

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, TensorDataset


def add_bias_column(t):
    return torch.cat([t, torch.ones([t.shape[0], 1])], 1)


def generate_toy_data():
        
    # source domain
    n                           = 1000
    mu                          = [0, 0]
    cov                         = [[20, -3], [-3, 4]]
    _X_source                   = np.random.multivariate_normal(mu, cov, n)
    _X_source                   = torch.tensor(_X_source, dtype = torch.float32)

    # Assign labels based on the position of data points relative to the first principal component
    _, _, _Vt                   = torch.linalg.svd(_X_source, full_matrices = False)
    first_pc                    = _Vt[0, :]
    _Y_source                   = torch.where(_X_source @ first_pc > 0, 1.0, -1.0)
    _Y_source                   = _Y_source.reshape([n, 1])

    # target distribution: rotate source domain by 45 degrees
    _deg                        = (np.random.choice(range(1,12), 1) * 15)[0]
    theta                       = np.radians(_deg)
    c, s                        = np.cos(theta), np.sin(theta) # Create a rotation matrix for 45 degrees
    rotation_matrix             = torch.tensor([[c, -s], [s, c]], dtype = torch.float32)
    _X_target                   = (rotation_matrix @ _X_source.T).T

    return add_bias_column(_X_source), _Y_source, add_bias_column(_X_target)


def get_svd(_X_source, _X_target):

    # SVD of source domain
    U, S, Vt                        = torch.linalg.svd(_X_source, full_matrices = False)
    S                               = S.reshape([S.shape[0],1]) # so we are prudent with vector dimensions
    
    # SVD of target domain
    U_tar, S_tar, Vt_tar            = torch.linalg.svd(_X_target, full_matrices = False)
    S_tar                           = S_tar.reshape([S_tar.shape[0],1])

    return U, S, Vt, U_tar, S_tar, Vt_tar


def check_label_mostly_in_span_of_top_singular_vectors(_Y_source, U):
    return U.T @ (_Y_source / torch.linalg.norm(_Y_source))


def custom_loss(y_pred, y_true, S, Vt, S_tar, Vt_tar, W, k, REGULARIZE = True, lam = 10.0):
    ret = torch.sum(torch.square(y_pred - y_true))
    if REGULARIZE:
        # note: good results w lam = 10. better results w lam = 1 and excluding align_s
        label_align_s = torch.sum(torch.square((S * (Vt @ W))[k:,:]))
        label_align_t = torch.sum(torch.square((S_tar * (Vt_tar @ W))[k:,:]))
        ret = ret - label_align_s + lam * label_align_t 
    return ret


class LinearClassifier(nn.Module): # logistic reg, with tanh activation for 1/-1 labels
    def __init__(self, width):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(width, 1, bias = False)
        self.linear.weight.data.fill_(0) # paper requires init to zero
        
    def forward(self, x):
        return torch.tanh(self.linear(x))


########################### recreating the toy example ######################################
#############################################################################################
#############################################################################################
#############################################################################################


if False:
        
    _X_source, _Y_source, _X_target = generate_toy_data()

    U, S, Vt, U_tar, S_tar, Vt_tar = get_svd(_X_source, _X_target)

    check_label_mostly_in_span_of_top_singular_vectors(_Y_source, U)

    # Create a model and train it
    epochs          = 100
    model           = LinearClassifier(_X_source.shape[1])
    optimizer       = optim.Adam(model.parameters(), lr = 0.1) # SGD doesnt work?
    losses          = []
    for epoch in range(epochs):
        # Forward pass
        predictions = model(_X_source)
        W           = model.linear.weight.reshape([-1,1])
        loss        = custom_loss(predictions, _Y_source, S, Vt, S_tar, Vt_tar, W, k=1, REGULARIZE=True)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}], Loss: {loss.item():.4f}')


    if True:
            
        x1_min, x1_max = _X_source[:, 0].min() - 1, _X_source[:, 0].max() + 1
        x2_min, x2_max = _X_source[:, 1].min() - 1, _X_source[:, 1].max() + 1
        x1_values      = np.linspace(x1_min, x1_max, 100)
        W              = model.linear.weight.detach().numpy()[0]
        x2_values      = -(W[0] * x1_values + W[2]) / W[1]

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
        _ = axes[0].plot(losses)
        _ = axes[1].scatter(_X_source[:, 0], _X_source[:, 1], c=_Y_source, cmap='viridis')
        _ = axes[1].plot(x1_values, x2_values, color='red')
        _ = axes[1].axis('equal')
        _ = axes[1].set_ylim(x2_min, x2_max)

        _ = axes[2].scatter(_X_target[:, 0], _X_target[:, 1], c=_Y_source, cmap='viridis')
        _ = axes[2].plot(x1_values, x2_values, color='red')
        _ = axes[2].axis('equal')
        _ = axes[2].set_ylim(x2_min, x2_max)

        _ = plt.tight_layout()
        _ = plt.show()


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
        


if False: # plotting for dataloader
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
    #     plt.show()

    # tr_usps, te_usps = get_data('USPS', False)
    # tr_mnst, te_mnst = get_data('MNIST', False)
    # plot_images(tr_usps)
    # plot_images(tr_mnst)
    # for i, (images, labels) in enumerate(tr_usps):
    #     print(images.shape)
    #     print(labels.shape)
    #     break


################ experiment 1: linear vs linear + regularizer ###############################
#############################################################################################
#############################################################################################
#############################################################################################


if False:

    scores = torch.zeros([90,3])
    i = 0
    for digit_1 in range(10):
        for digit_2 in [i for i in range(10) if i != digit_1]:
            digits = [digit_1, digit_2]
            print(digits)




            xtr, ytr, xte, yte                = get_data('MNIST', digits)
            _X_source                         = add_bias_column(xtr.reshape([-1, 784]))
            _X_target                         = add_bias_column(xte.reshape([-1, 784]))
            _Y_source                         = ytr.reshape([ytr.shape[0], 1])

            # check dims, otherwise SVD will not have 784 dims
            assert _X_source.shape[0] > _X_source.shape[1], '_X_source not enough data points'
            assert _X_target.shape[0] > _X_target.shape[1], '_X_source not enough data points'

            # convert labels to 1/-1 for the label align property, or 0/1 for logistic
            _Y_source[_Y_source == digits[0]] = -1
            _Y_source[_Y_source == digits[1]] = 1
            _Y_source                         = _Y_source.float()

            U, S, Vt, U_tar, S_tar, Vt_tar = get_svd(_X_source, _X_target)
            # U, S, Vt, U_tar, S_tar, Vt_tar = None, None, None, None, None, None, 

            # check_label_mostly_in_span_of_top_singular_vectors(_Y_source, U)[:10].flatten()

            # Create a model and train it
            epochs          = 1000
            model           = LinearClassifier(_X_source.shape[1])
            optimizer       = optim.Adam(model.parameters(), lr = 0.1) # SGD doesnt work?
            # losses          = []
            for epoch in range(epochs):
                # Forward pass
                predictions = model(_X_source)
                W           = model.linear.weight.reshape([-1,1])
                loss        = custom_loss(predictions, _Y_source, S, Vt, S_tar, Vt_tar, W, k=3, REGULARIZE=True, lam = 1000.0)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # losses.append(loss.item())
                # if (epoch + 1) % 10 == 0:
                #     print(f'Epoch [{epoch + 1}], Loss: {loss.item():.4f}')

            # evaluate logistic regression on source domain's test set 
            # xte                          = xte.reshape([-1, 784])
            # _labels                      = torch.zeros([yte.shape[0],1])
            # _labels[yte == digits[1],:]  = 1
            # _pred                        = torch.zeros([xte.shape[0],1])
            # _pred[model(xte) > 0.5]      = 1
            # _accuracy                    = torch.sum(_pred == _labels) / xte.shape[0]
            # _accuracy

            # evaluate logistic regression on target domain
            _xte1, _yte1, _xte2, _yte2   = get_data('USPS', digits)
            xte2                         = torch.cat([_xte1, _xte2], 0)
            yte2                         = torch.cat([_yte1, _yte2], 0)
            xte2                         = add_bias_column(xte2.reshape([-1, 784]))
            _labels                      = torch.zeros([yte2.shape[0],1]) -1
            _pred                        = torch.zeros([xte2.shape[0],1]) -1
            _labels[yte2 == digits[1],:] = 1
            _pred[model(xte2) > 0.5]     = 1
            _accuracy                    = torch.sum(_pred == _labels) / xte2.shape[0]
            _accuracy

            scores[i,:]                  = torch.tensor([digit_1, digit_2, _accuracy])
            i                            = i + 1


    # # this is a code snippet I used to choose k = 3 for this problem. 
    # U, S, Vt, U_tar, S_tar, Vt_tar = get_svd(_X_source, _X_target)
    # components = U.T @ (_Y_source / torch.norm(_Y_source))
    # components[:10].flatten()

    # plt.plot(torch.abs(components)[:4])
    # plt.show()


    '''
    # results of basic logistic regression
    >>> scores
    tensor([[0.0000, 1.0000, 0.9784],
            [0.0000, 2.0000, 0.4944],
            [0.0000, 3.0000, 0.3589],
            [0.0000, 4.0000, 0.9293],
            [0.0000, 5.0000, 0.6580],
            [0.0000, 6.0000, 0.7507],
            [0.0000, 7.0000, 0.6610],
            [0.0000, 8.0000, 0.7802],
            [0.0000, 9.0000, 0.8142],
            [1.0000, 0.0000, 0.9780],
            [1.0000, 2.0000, 0.9240],
            [1.0000, 3.0000, 0.9345],
            [1.0000, 4.0000, 0.9670],
            [1.0000, 5.0000, 0.8353],
            [1.0000, 6.0000, 0.7009],
            [1.0000, 7.0000, 0.8937],
            [1.0000, 8.0000, 0.8730],
            [1.0000, 9.0000, 0.9167],
            [2.0000, 0.0000, 0.4879],
            [2.0000, 1.0000, 0.9231],
            [2.0000, 3.0000, 0.7262],
            [2.0000, 4.0000, 0.8731],
            [2.0000, 5.0000, 0.7872],
            [2.0000, 6.0000, 0.5695],
            [2.0000, 7.0000, 0.7374],
            [2.0000, 8.0000, 0.8668],
            [2.0000, 9.0000, 0.6651],
            [3.0000, 0.0000, 0.3589],
            [3.0000, 1.0000, 0.9345],
            [3.0000, 2.0000, 0.7268],
            [3.0000, 4.0000, 0.8484],
            [3.0000, 5.0000, 0.5760],
            [3.0000, 6.0000, 0.5332],
            [3.0000, 7.0000, 0.6411],
            [3.0000, 8.0000, 0.5836],
            [3.0000, 9.0000, 0.5617],
            [4.0000, 0.0000, 0.9293],
            [4.0000, 1.0000, 0.9689],
            [4.0000, 2.0000, 0.8731],
            [4.0000, 3.0000, 0.8490],
            [4.0000, 5.0000, 0.8272],
            [4.0000, 6.0000, 0.5991],
            [4.0000, 7.0000, 0.5474],
            [4.0000, 8.0000, 0.8596],
            [4.0000, 9.0000, 0.6115],
            [5.0000, 0.0000, 0.6571],
            [5.0000, 1.0000, 0.8383],
            [5.0000, 2.0000, 0.7878],
            [5.0000, 3.0000, 0.5766],
            [5.0000, 4.0000, 0.8297],
            [5.0000, 6.0000, 0.4213],
            [5.0000, 7.0000, 0.6412],
            [5.0000, 8.0000, 0.6229],
            [5.0000, 9.0000, 0.6096],
            [6.0000, 0.0000, 0.7541],
            [6.0000, 1.0000, 0.7047],
            [6.0000, 2.0000, 0.5701],
            [6.0000, 3.0000, 0.5356],
            [6.0000, 4.0000, 0.6074],
            [6.0000, 5.0000, 0.4206],
            [6.0000, 7.0000, 0.3592],
            [6.0000, 8.0000, 0.7542],
            [6.0000, 9.0000, 0.5625],
            [7.0000, 0.0000, 0.6571],
            [7.0000, 1.0000, 0.9000],
            [7.0000, 2.0000, 0.7443],
            [7.0000, 3.0000, 0.6473],
            [7.0000, 4.0000, 0.5517],
            [7.0000, 5.0000, 0.6426],
            [7.0000, 6.0000, 0.3555],
            [7.0000, 8.0000, 0.6880],
            [7.0000, 9.0000, 0.5728],
            [8.0000, 0.0000, 0.7762],
            [8.0000, 1.0000, 0.8751],
            [8.0000, 2.0000, 0.8748],
            [8.0000, 3.0000, 0.5849],
            [8.0000, 4.0000, 0.8628],
            [8.0000, 5.0000, 0.6194],
            [8.0000, 6.0000, 0.7497],
            [8.0000, 7.0000, 0.6933],
            [8.0000, 9.0000, 0.7508],
            [9.0000, 0.0000, 0.8100],
            [9.0000, 1.0000, 0.9187],
            [9.0000, 2.0000, 0.6731],
            [9.0000, 3.0000, 0.5666],
            [9.0000, 4.0000, 0.6115],
            [9.0000, 5.0000, 0.6142],
            [9.0000, 6.0000, 0.5577],
            [9.0000, 7.0000, 0.5759],
            [9.0000, 8.0000, 0.7567]])

    >>> torch.mean(scores[:,2])
    tensor(0.7089)
    '''

    '''
    results of using the regularization

    >>> scores
    tensor([[0.0000, 1.0000, 0.3519],
            [0.0000, 2.0000, 0.4601],
            [0.0000, 3.0000, 0.8772],
            [0.0000, 4.0000, 0.6802],
            [0.0000, 5.0000, 0.7876],
            [0.0000, 6.0000, 0.6850],
            [0.0000, 7.0000, 0.9249],
            [0.0000, 8.0000, 0.9341],
            [0.0000, 9.0000, 0.9511],
            [1.0000, 0.0000, 0.3512],
            [1.0000, 2.0000, 0.9145],
            [1.0000, 3.0000, 0.9212],
            [1.0000, 4.0000, 0.9349],
            [1.0000, 5.0000, 0.9189],
            [1.0000, 6.0000, 0.7594],
            [1.0000, 7.0000, 0.9457],
            [1.0000, 8.0000, 0.8624],
            [1.0000, 9.0000, 0.9139],
            [2.0000, 0.0000, 0.4517],
            [2.0000, 1.0000, 0.9240],
            [2.0000, 3.0000, 0.3691],
            [2.0000, 4.0000, 0.8097],
            [2.0000, 5.0000, 0.4365],
            [2.0000, 6.0000, 0.8332],
            [2.0000, 7.0000, 0.6944],
            [2.0000, 8.0000, 0.7465],
            [2.0000, 9.0000, 0.7029],
            [3.0000, 0.0000, 0.8763],
            [3.0000, 1.0000, 0.9269],
            [3.0000, 2.0000, 0.3697],
            [3.0000, 4.0000, 0.8144],
            [3.0000, 5.0000, 0.6143],
            [3.0000, 6.0000, 0.6375],
            [3.0000, 7.0000, 0.7933],
            [3.0000, 8.0000, 0.7937],
            [3.0000, 9.0000, 0.7295],
            [4.0000, 0.0000, 0.6707],
            [4.0000, 1.0000, 0.9392],
            [4.0000, 2.0000, 0.8097],
            [4.0000, 3.0000, 0.8103],
            [4.0000, 5.0000, 0.7985],
            [4.0000, 6.0000, 0.5824],
            [4.0000, 7.0000, 0.5547],
            [4.0000, 8.0000, 0.7853],
            [4.0000, 9.0000, 0.6426],
            [5.0000, 0.0000, 0.7880],
            [5.0000, 1.0000, 0.9144],
            [5.0000, 2.0000, 0.4371],
            [5.0000, 3.0000, 0.6123],
            [5.0000, 4.0000, 0.7985],
            [5.0000, 6.0000, 0.6097],
            [5.0000, 7.0000, 0.5776],
            [5.0000, 8.0000, 0.5836],
            [5.0000, 9.0000, 0.7476],
            [6.0000, 0.0000, 0.6850],
            [6.0000, 1.0000, 0.7451],
            [6.0000, 2.0000, 0.8332],
            [6.0000, 3.0000, 0.6393],
            [6.0000, 4.0000, 0.5836],
            [6.0000, 5.0000, 0.6135],
            [6.0000, 7.0000, 0.6230],
            [6.0000, 8.0000, 0.6907],
            [6.0000, 9.0000, 0.7789],
            [7.0000, 0.0000, 0.9254],
            [7.0000, 1.0000, 0.9423],
            [7.0000, 2.0000, 0.6961],
            [7.0000, 3.0000, 0.7921],
            [7.0000, 4.0000, 0.5547],
            [7.0000, 5.0000, 0.5769],
            [7.0000, 6.0000, 0.6199],
            [7.0000, 8.0000, 0.7600],
            [7.0000, 9.0000, 0.6020],
            [8.0000, 0.0000, 0.9275],
            [8.0000, 1.0000, 0.8609],
            [8.0000, 2.0000, 0.7465],
            [8.0000, 3.0000, 0.7918],
            [8.0000, 4.0000, 0.7853],
            [8.0000, 5.0000, 0.5836],
            [8.0000, 6.0000, 0.6900],
            [8.0000, 7.0000, 0.7613],
            [8.0000, 9.0000, 0.7417],
            [9.0000, 0.0000, 0.9499],
            [9.0000, 1.0000, 0.9081],
            [9.0000, 2.0000, 0.7029],
            [9.0000, 3.0000, 0.7410],
            [9.0000, 4.0000, 0.6408],
            [9.0000, 5.0000, 0.7450],
            [9.0000, 6.0000, 0.7789],
            [9.0000, 7.0000, 0.6020],
            [9.0000, 8.0000, 0.7417]])

    >>> torch.mean(scores[:,2])
    tensor(0.7302)

    fairly similar result to what the paper claimed, about 2-3% improvement.
    '''


##### try one-hot encoded classes, see if the one-hot vecs are in span of singular vecs #####
############################################################### answer: they are! ###########
#############################################################################################
#############################################################################################



digits = range(10)

xtr, ytr, xte, yte                = get_data('MNIST', digits)
_X_source                         = add_bias_column(xtr.reshape([-1, 784]))
_X_target                         = add_bias_column(xte.reshape([-1, 784]))
_Y_source                         = ytr.reshape([ytr.shape[0], 1])
_Y_source                         = _Y_source.float()

# check dims, otherwise SVD will not have 784 dims
assert _X_source.shape[0] > _X_source.shape[1], '_X_source not enough data points'
assert _X_target.shape[0] > _X_target.shape[1], '_X_source not enough data points'

# convert labels to 1/-1 for the label align property
labels = torch.zeros([_Y_source.shape[0], len(digits)]) - 1
for i in range(len(digits)):
    labels[(_Y_source == digits[i]).flatten(),i] = 1

U, S, Vt, U_tar, S_tar, Vt_tar = get_svd(_X_source, _X_target)

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


class LinearClassifier(nn.Module): # logistic reg, with tanh activation for 1/-1 labels
    def __init__(self, in_dim, out_dim):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias = False)
        # self.linear.weight.data.fill_(0) # paper requires init to zero
        
    def forward(self, x):
        return torch.softmax(self.linear(x), dim=1) * 2 - 1


Y_tr = torch.zeros([_X_source.shape[0], len(digits)]) - 1
Y_tr[torch.arange(_X_source.shape[0]), _Y_source.to(torch.long).flatten()] = 1


# Create a model and train it
epochs          = 1000
model           = LinearClassifier(_X_source.shape[1], len(digits))
optimizer       = optim.Adam(model.parameters()) # SGD doesnt work?
losses          = []
for epoch in range(epochs):
    # Forward pass
    predictions = model(_X_source)

    W           = model.linear.weight.T
    # loss        = custom_loss(predictions, Y_tr, S, Vt, S_tar, Vt_tar, W, k=3, REGULARIZE=False)
    loss        = custom_loss(predictions, Y_tr, S, Vt, S_tar, Vt_tar, W, k=3, REGULARIZE=True, lam = 1000.0)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}], Loss: {loss.item():.4f}')


def test(x, y, model):
    p                             = torch.argmax(model(x), dim = 1)
    acc                           = torch.sum(p == y) / x.shape[0]
    return acc

# evaluate logistic regression on source domain's test set 
_xte                              = add_bias_column(xte.reshape([-1, 784]))
test(_xte, yte, model)

# evaluate logistic regression on target domain
__xte1, __yte1, __xte2, __yte2    = get_data('USPS', digits)
_xte                              = torch.cat([__xte1, __xte2], 0)
_yte                              = torch.cat([__yte1, __yte2], 0)
_xte                              = add_bias_column(_xte.reshape([-1, 784]))
test(_xte, _yte, model)

# BASELINE: 
# linear regression gets 93 % on source test set, 23 % on target set
# by including the regularizer, yield 46 % source, 41 % target
# removing the w=0 requirement increases target performance form 31% to 40%
# having lr=0.1 made the accuracy vary from 15 to 40 %, removing it made it 40 % reliably





####################### P E R T U R B A T I O N  ############################################
#############################################################################################
#############################################################################################
#############################################################################################

xtr, ytr, xte, yte                = get_data('MNIST', digits)
_X_source                         = add_bias_column(xtr.reshape([-1, 784]))
_X_target                         = add_bias_column(xte.reshape([-1, 784]))
_Y_source                         = ytr.reshape([ytr.shape[0], 1])
_Y_source                         = _Y_source.float()

# check dims, otherwise SVD will not have 784 dims
assert _X_source.shape[0] > _X_source.shape[1], '_X_source not enough data points'
assert _X_target.shape[0] > _X_target.shape[1], '_X_source not enough data points'

# convert labels to 1/-1 for the label align property
labels = torch.zeros([_Y_source.shape[0], len(digits)]) - 1
for i in range(len(digits)):
    labels[(_Y_source == digits[i]).flatten(),i] = 1



A = torch.cat([_X_source, torch.zeros([10000,785])], dim = 0) # extend with 10k zeros (where next points will go)
U, S, Vt, U_, S_, Vt_ = get_svd(_X_source, A)

S[:10].flatten()
S_[:10].flatten()


xte = add_bias_column(xte.reshape([10000, 784]))
Ap = torch.cat([_X_source, xte], dim = 0) # added 10k data points from test set 


'''

- - - things that affect sample complexity
    - model complexity (nn vs linear reg) (how complex is the decision boundary we need to learn?)
    - noise / outliers 
    - difficulty of task: faces vs digits, faces have more complex structure
    - dimensionality of data
    - underlying structure of data can be in a low dimensional manifold

- does sample complexity apply to problems where the label vector aligns with the top singular vectors of the data matrix?
- Yes. In this case, the sample complexity can be lower than a general case, because alignment of the label vector with the top singular vectors indicated that the most important information for predicting labels is mostly in a low-dimensional subspace. This can simplify the learning objective and allow the model to learn the underlying structure with fewer samples. We must set some other variables constant such as the model architecture, the optimization algorithm (Adam), and the hyperparameters. There is no way to measure complexity and give a number, but we can compare two problems as being more or less complex. 

    - generate example data sets to measure relative performance in different situations
TODO: example datasets at 
"Sure, we can generate example datasets based on the factors I mentioned "


- for our case, we are investigating problems that have the alignment property. 
    generate synthetics datasets that exhibit these properties. measure performance of models:

    - Alignment strength: how much of label norm is captured in projection. more norm => less data?
    > measure projection norm, good if it's closer to 1 

    - Number of significant singular vectors: majority of projected norm in few vecs => less data?
    > how many vecs contain most of the projection norm?

    - Stability of singular vectors: do singular vecs change significantly when we add new data, or data distribution shifts?
    > see below

    
- - - stability of singular vectors 
    - When you add new data points or the data distribution shifts, the singular vectors can change. 
    - stability of singular vectors is related to the geometry of the data, as well as the nature of the underlying distribution. 
        One way to think about it is that when the data is well-distributed and the underlying structure is robust, 
        the singular vectors will be more stable.

    - The row and column spaces of the data matrix are indeed related to the stability of singular vectors. 
        When the row space and column space are stable, the singular vectors will be more stable as well.


    - if the singular values decay slowly, it indicates that the singular vectors corresponding to 
        the smaller singular values still capture some useful information about the data, and perturbations in the data matrix can 
        significantly affect the singular vectors. 
        
    - if the singular values decay rapidly, the singular vectors corresponding to the smaller singular values capture less information 
        about the data, and perturbations in the data matrix are less likely to affect the singular vectors significantly.

    - to make sure the singular vectors remain stable, it's important to have a good representation of the underlying distribution of the data. 
        If the data is sampled from the same distribution, even when new data points are added, the row and column spaces will remain 
        relatively stable, leading to more stable singular vectors. 
        Moreover, pre-processing techniques, such as normalization and feature scaling, can improve the stability of singular vectors.

    - One way to analyze the stability of singular vectors is to look at the perturbation theory for singular value decomposition (SVD). 
        it examines how small changes in the data matrix affect the singular vectors and singular values. 

        This can be done by examining the bounds on the changes in singular values and vectors in response to matrix perturbations. 
        A key reference for studying perturbation theory for SVD is the book "Matrix Computations" by Gene H. Golub and Charles F. Van Loan.


    - if the data is spread out along some line in space, the singular vector in that direction will likely be more stable. 
        This is because the dominant singular vector captures the direction of maximal variance, 
        and if the data is well-distributed along a line, the variance in that direction will be consistently high.

    - In the case of data that is spherical along a subspace, the singular vectors may indeed be less stable. 
        If the data is uniformly distributed within the sphere, any rotation of the singular vectors will still represent the same subspace. 
        In this case, the singular vectors are not uniquely determined and can be affected by small perturbations in the data.

    - see Steve Brunton for singular value thresholding, related to ideas from information theory, such as entropy and mutual information.

    - most real-world datasets are not perfectly linear. In these cases, you can still use SVD to find the best linear approximation 
        of the data by projecting it onto a lower-dimensional subspace. This can be helpful for tasks like dimensionality reduction, 
        data compression, and noise reduction.



When discussing bounds on the changes in singular values and vectors due to matrix perturbations, we typically refer to both left and right singular vectors. Perturbation analysis helps us understand how much the singular values and vectors can change when the input matrix is altered slightly. In many practical situations, data is subject to small perturbations, such as noise or errors in measurements. Understanding the behavior of the SVD under these perturbations can be important for the stability of algorithms that rely on SVD.

The singular value decomposition of a matrix A is given by:

A = UΣV^T

where U and V are unitary matrices containing the left and right singular vectors, respectively, and Σ is a diagonal matrix containing the singular values.

Let's consider a perturbed matrix A + E, where E is a small perturbation matrix. The perturbed matrix has the SVD:

A + E = U'Σ'V'^T

Here, U' and V' are the perturbed left and right singular vectors, and Σ' is the perturbed singular values.

One useful result from perturbation analysis is the bounds on the changes in singular values. For any i, the difference between the original and perturbed singular values is bounded by the matrix norm of the perturbation:

|σ_i - σ'_i| ≤ ||E||

Similarly, the bounds on the changes in left and right singular vectors can be expressed using the concept of canonical angles. Let θ_ij be the angle between the i-th original left (or right) singular vector and the j-th perturbed left (or right) singular vector. Then, the sine of the canonical angles can be bounded by the matrix norm of the perturbation:

sin(θ_ij) ≤ ||E|| / σ_i

These bounds give us an understanding of how much the singular values and vectors can change due to small perturbations in the matrix. In general, the left and right singular vectors are more sensitive to perturbations when the corresponding singular values are small. This is because small singular values indicate that the matrix is nearly singular or rank-deficient, making the associated singular vectors less stable under perturbations.


- - - how can adding data to the data matrix be related to the norm of the pterturbation?


When you add additional data points to the data matrix, you are essentially perturbing the matrix by extending its row space. Let's consider the following scenario:

You have a data matrix A of size n x d, where n is the number of data points and d is the dimensionality of each point.
You add m new data points, resulting in an extended data matrix A' of size (n+m) x d.
To analyze the perturbation due to adding new data points, you can create a perturbation matrix E of size (n+m) x d as follows:

The first n rows of E are all zeros, matching the original rows of A.
The last m rows of E contain the new data points.
In this case, the perturbed matrix A + E is equal to the extended data matrix A'. The matrix norm of E, ||E||, represents the "magnitude" of the perturbation caused by adding the new data points.

However, this approach does not capture the changes in the singular vectors directly because the perturbation matrix E has a different row space than A. Instead, you can compute the SVD of the extended data matrix A' and compare its singular values and vectors with those of the original matrix A to understand the effects of adding new data points.

As you add more data points, the singular values and vectors of A' may change. In general, if the new data points are aligned with the existing structure in the data, the dominant singular values and vectors will remain stable, and the corresponding subspace will be more robust to the addition of new data points. On the other hand, if the new data points introduce new structures or directions in the data, the singular values and vectors may change significantly. The extent to which the SVD is affected by adding new data points depends on the geometry of the data and the relationships between the old and new data points.


- - - Can't we use the right singular vectors to compare A and A', since the right singular vectors of both matrices will have the same dimensions d by d?


Yes, you can compare the right singular vectors of A and A' as they both have the same dimensions (d x d). This comparison will give you an idea of how the addition of new data points has affected the relationships between the features (dimensions) in the data.

To do this, you can compute the SVD of both A and A', and then compare their right singular vectors V and V':

Compute the SVD of A, resulting in U_A, S_A, and V_A.
Compute the SVD of A', resulting in U_A', S_A', and V_A'.
Compare the right singular vectors V_A and V_A' by calculating the difference, e.g., ||V_A - V_A'||, or by computing the angle between the corresponding right singular vectors.
Keep in mind that comparing right singular vectors is focused on the relationships between features (dimensions) in the data, rather than the relationships between data points. However, if the addition of new data points significantly changes the structure in the data, you will likely observe differences in the right singular vectors as well.

It's important to note that the stability of the right singular vectors is related to the stability of the left singular vectors, as the left and right singular vectors are connected through the singular values in the SVD. If the dominant left singular vectors remain stable, it is likely that the corresponding right singular vectors will also remain stable.


- - - What exactly is the relation between the stability of the left singular vectors and stability of the right singular vectors? Please include any mathematical formulas, if they help.


The stability of the left singular vectors and right singular vectors is connected through the singular values in the SVD. Recall that the SVD of a matrix A is given by A = UΣV^T, where U and V are orthogonal matrices containing the left and right singular vectors, respectively, and Σ is a diagonal matrix containing the singular values.

When a matrix A is perturbed by a small change ΔA, the perturbed matrix A' = A + ΔA also has an SVD, given by A' = U'Σ'V'^T. To understand the relation between the stability of the left and right singular vectors, we can examine how the singular values and singular vectors change as a result of this perturbation.

The Davis-Kahan theorem provides a bound on the difference between the singular subspaces of A and A' that is related to the perturbation size and the separation between the singular values. For a pair of singular values σ_i and σ_{i+1}, let μ = min{|σ_i - σ_j| : j ≠ i}. Then, the theorem states that:

||sin(Θ(U_i, U'_i))|| ≤ ||ΔA|| / μ
||sin(Θ(V_i, V'_i))|| ≤ ||ΔA|| / μ

Here, Θ(U_i, U'_i) and Θ(V_i, V'_i) denote the angles between the i-th left and right singular vector pairs of A and A', respectively. The theorem shows that the difference between the singular vector pairs is bounded by the perturbation size ||ΔA|| and inversely proportional to the separation μ between the singular values.

From this theorem, we can conclude that if the singular values are well-separated (i.e., μ is large), the corresponding left and right singular vectors will be relatively stable under small perturbations. Conversely, if the singular values are close together, the left and right singular vectors may change significantly even for small perturbations.

In summary, the stability of the left singular vectors and the right singular vectors is related through the singular values. If the dominant singular values are well-separated, both the left and right singular vectors corresponding to those singular values will be relatively stable under small perturbations.


- - - How does data that is not perfectly linear tie into all this? will it have less stable singular vectors, since new data points are likely to perturb the structure of the data? Also, what is a good synthetic data set we can construct to study the effect of perturbations from additional data? what kind of analysis should we perform on the singular vectors?


Data that is not perfectly linear will generally have less stable singular vectors than perfectly linear data because the spread of the data is not concentrated along a single direction. New data points can indeed perturb the structure of the data, affecting the singular values and vectors. However, if the data still exhibits a strong underlying structure, the most dominant singular vectors might remain relatively stable.

To construct a synthetic dataset to study the effect of perturbations from additional data, you can create a dataset that is initially linear and then add varying levels of noise. Here's an example of how to create such a dataset:

Generate a linear dataset by sampling points along a line.
Add Gaussian noise to the data points, gradually increasing the variance of the noise to create different levels of perturbation.
Here's a code example using numpy to generate such a dataset:

import numpy as np
import matplotlib.pyplot as plt

def angle_between_vectors(u, v):
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    return np.degrees(theta)

n_points = 100
noise_levels = [0, 0.1, 0.5, 1]

x = np.linspace(0, 10, n_points)
line = 2 * x + 3

fig, axes = plt.subplots(1, len(noise_levels), figsize=(20, 5))

for i, noise_level in enumerate(noise_levels):
    noise = np.random.normal(0, noise_level, n_points)
    perturbed_data = line + noise
    
    axes[i].scatter(x, perturbed_data)
    axes[i].set_title(f'Noise Level: {noise_level}')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('y')

plt.show()


To analyze the effect of perturbations on the singular vectors, you can perform the following steps:

Compute the SVD for each dataset (with different noise levels).
Observe the singular values and how they change with increasing noise levels.
Observe the angles between the dominant left and right singular vectors for each dataset and the original linear dataset.
You can calculate the angle between two singular vectors using the dot product and arccosine function:

By analyzing how the angles between the dominant singular vectors change with increasing noise levels, you can gain insights into the stability of the singular vectors under different levels of perturbation. If the angles remain small despite the noise, the singular vectors are relatively stable. Conversely, if the angles increase significantly, the singular vectors are less stable.


'''


# PRIORITIES:
# # 2. read the other paper 


# TODO: is taking SVD of n by d matrix the same as taking SVD of infinity by d matrix, where rows n+1...infty are all zero?

# TODO: how does having a source and target density tie into all of this?

'''
if the data has columns that are highly correlated to the label vec, then the label alignment property emerges. 

lemma 3
if the k right singular vectors satisfy ||X v_i|| < e, then X has at most d-k singular values > sqrt(k)*e

prop 4
if X has k columns which dotted with Y have dot product (corr) > 0.8, 
'''

#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################

# - - - 680 project road map 
#     x - implement toy example w regularization
#     x - implement MNIST vs USPS w regularization
#     
#     - - - extending to multi-class
#     x - check that one-hot vectors have label alignment property
#     x - extend MNIST vs USPS to multiclas
#     o - extend proof in appendix A to multiclass 
#
#     - - - sample complexity
#     o - go over the sample complexity discussion & see what i can do 


# As for your question about whether sample complexity can apply to your problem where the label vector aligns with the top singular vectors of the data matrix, the answer is yes. In this case, the sample complexity could be lower than in a more general scenario, as the alignment of the label vector with the top singular vectors indicates that the most important information for predicting the labels is concentrated in a low-dimensional subspace. This simplifies the learning task and could allow the model to achieve good performance with fewer samples.

# toy example 1: 2D data rotated in n-dim space. 

# what makes a problem more complex? How does a problem become more complex, ie require more data?

# probably a larger dimension subspace = more samples (generate toy example to show this)
# also, probably the linearity? or maybe put a NN in previous layer to make the features linear? 

# i can try variant example datasets to study how many samples are needed to get a certain level of accuracy:
# compare: increase/decrease dimensionality of subspace 
# compare: high vs low dimensionality (keeping the subspace dimension constant)
# compare: linear model vs nn basis function (also whats up with hidden representations? see 1st paper)
# compare: more/less noise














DejaVu
Symbola_hint
NotoSerif-Regular
NotoSansMono-Medium
NotoSansDisplay-Regular
















# using PyGame instead of tkinter because tkinter renders fonts pixelated

import subprocess 
import pygame

# get a list of the system fonts
# :lang=el                                  <- fonts that support unicode Greek (for math) 
# grep -viE 'italic|bold|condensed|oblique' <- this exclude bold, italic, etc fonts
# cut -d ':' -f 1                           <- this takes out just the file name
# sort -u                                   <- this sorts and removes duplicates
cmd = "fc-list :lang=el | grep -viE 'italic|bold|condensed|oblique' | cut -d ':' -f 1 | sort -u"
# cmd = "fc-list :lang=el | grep -viE 'italic|bold|condensed|oblique' | grep 'DejaVu' | cut -d ':' -f 1 | sort -u"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
font_files = [i for i in result.stdout.split('\n') if len(i) > 0]

# the lines to show. I think it's better to actually render something you'd be reading.
lines = []
lines.append('To find the maximum likelihood estimate (MLE) of θ²,')
lines.append('we first need to find the likelihood function, ')
lines.append('then its derivative with respect to θ, ')
lines.append('and finally set the derivative equal to 0 and solve for θ.')
lines.append('Since X₁, ..., Xₙ are independent and identically distributed (i.i.d.) random variables, ')
lines.append('the likelihood function L(θ) is given by the product of probabilities:')
lines.append('L(θ) = P(X₁ = x₁, ..., Xₙ = xₙ | θ) = Πᵢ P(Xᵢ = xᵢ | θ)')
lines.append('Given that Pr(Xᵢ = 1) = θ and Pr(Xᵢ = 0) = 1 - θ, the likelihood function can be written as:')
lines.append('L(θ) = θ^s * (1 - θ)^(n - s)')
lines.append("where s is the number of successes (i.e., the number of 1's) in the sample.")
lines.append('')




# Initialize Pygame
pygame.init()

# Set the screen size
screen_width = 1200
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))

# Define the font size and sample text
font_size = 24

# Load the font files and create a list of font objects
fonts = []
for font_file in font_files:
    font = pygame.font.Font(font_file, font_size)
    fonts.append(font)

# Set the initial font index to 0
font_index = 0

# Run the main loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                # Switch to the next font
                font_index = (font_index + 1) % len(fonts)
            elif event.key == pygame.K_LEFT:
                # Switch to the previous font
                font_index = (font_index - 1) % len(fonts)
            elif event.key == pygame.K_ESCAPE:
                # Quit the application and clean up resources
                running = False
                pygame.quit()

    # Create a new image with a white background
    _             = img = pygame.Surface((screen_width, screen_height))
    _             = img.fill((91,26,56))

    # Render the sample text using the current font
    font          = fonts[font_index]
    v_offset      = font_size
    lines.append(font_files[font_index]) # add the current font so we know the file name
    for line in lines:
        v_offset  = v_offset + font_size * 1.3
        text      = font.render(line, True, (225,204,140))
        text_rect = text.get_rect(topleft=(font_size, v_offset))
        _         = img.blit(text, text_rect)
    lines.pop() # remove current font file name

    # Update the display
    _             = screen.blit(img, (0, 0))
    _             = pygame.display.flip()

# Clean up
pygame.quit()














# function to copy text to clipboard (linux)
import subprocess
def copy_to_clipboard(text):
    p = subprocess.Popen(['xsel', '-bi'], stdin=subprocess.PIPE)
    p.communicate(input=text.encode())


'''



'''



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

