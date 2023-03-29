

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import PIL.Image as im
from torchvision import transforms, datasets


def gen_grid(size):
    lim = 0.5
    x = np.linspace(-lim, lim, size)
    y = np.linspace(-lim, lim, size)
    xx, yy = np.meshgrid(x,y)
    return torch.Tensor((yy.ravel(), xx.ravel())).T



# ======================================== B A S I C ==============================================
# ======================================== B A S I C ==============================================

if False:
        
    class NN(nn.Module):

        def __init__(self, w=16, num_layers = 9, activation = nn.Tanh):
            super(NN, self).__init__()

            layers = [nn.Linear(2, w), activation()]
            for _ in range(num_layers - 1):
                layers += [nn.Linear(w, w), activation()]
            layers += [nn.Linear(w, 3), nn.Sigmoid()]
            self.layers = nn.Sequential(*layers)
            for i in self.layers:
                if type(i) == nn.Linear:
                    nn.init.normal_(i.weight)

        def forward(self, x):
            return self.layers(x)

    size = 128 

    net = NN()
    grid = gen_grid(size)

    img = net(grid).detach().numpy()
    colors = (255.0 * img.reshape(size, size, 3)).astype(np.uint8)
    im.fromarray(colors).show()



# ======================================== S I N G L E   M N I S T ==============================================
# ======================================== S I N G L E   M N I S T ==============================================

if False:
        
    batch_size = 1
    DATA_DIR = r'C:\Users\i_hat\Desktop\losable\pytorch_data'

    train_loader = torch.utils.data.DataLoader(datasets.MNIST(DATA_DIR, train=True,
                        transform=transforms.Compose([transforms.ToTensor()])), batch_size = batch_size, shuffle=True, drop_last = True)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST(DATA_DIR, train=False,
                        transform=transforms.Compose([transforms.ToTensor()])), batch_size = batch_size, shuffle = True)


    class NN(nn.Module):

        def __init__(self, w=16, num_layers = 9, activation = nn.Tanh):
            super(NN, self).__init__()

            layers = [nn.Linear(3, w), activation()]
            for _ in range(num_layers - 1):
                layers += [nn.Linear(w, w), activation()]
            layers += [nn.Linear(w, 1), nn.Sigmoid()]
            self.layers = nn.Sequential(*layers)
            for i in self.layers:
                if type(i) == nn.Linear:
                    nn.init.normal_(i.weight)

        def forward(self, x):
            return self.layers(x)

    size = 28 
    grid = gen_grid(size)
    x, y = next(iter(train_loader))
    x = x.reshape([784,1])

    device = torch.device('cuda')
    device = torch.device('cpu')

    net = NN()
    net.to(device)
    _opt = torch.optim.Adam(net.parameters(), 0.001)#, weight_decay = 0.0)

    _in = torch.cat([grid, y[:,None].expand([784,1])],1)
    for i in range(5_000):
        _opt.zero_grad()
        _out = net(_in)
        _loss = F.mse_loss(_out, x)
        _loss.backward()
        _opt.step()


    size = 28
    grid = gen_grid(size)
    #   R E C O N 
    im.fromarray(np.uint8(255.0 * _out.detach().numpy().squeeze().reshape([28,28]))).show()
    #   O R I G   I M A G E 
    im.fromarray(np.uint8(255.0 * x.detach().numpy().squeeze().reshape([28,28]))).show()

    #   B I G   G R I D
    _label = y
    size = 1080
    grid = gen_grid(size)
    _in = torch.cat([grid, _label[:,None].expand([size*size,1])],1)
    im.fromarray(np.uint8(255.0 * net(_in).detach().numpy().squeeze().reshape([size,size]))).show()



# ======================================== M A N Y   M N I S T ==============================================
# ======================================== M A N Y   M N I S T ==============================================


import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import PIL.Image as im
from torchvision import transforms, datasets


def gen_grid(size):
    lim = 0.5
    x = np.linspace(-lim, lim, size)
    y = np.linspace(-lim, lim, size)
    xx, yy = np.meshgrid(x,y)
    return torch.Tensor((yy.ravel(), xx.ravel())).T

batch_size = 16
out_dir = r'C:\Users\i_hat\Desktop\bastl\py\deep_larn\CPPN'
DATA_DIR = r'C:\Users\i_hat\Desktop\losable\pytorch_data'

train_loader = torch.utils.data.DataLoader(datasets.MNIST(DATA_DIR, train=True,
                    transform=transforms.Compose([transforms.ToTensor()])), batch_size = batch_size, shuffle=True, drop_last = True)

test_loader = torch.utils.data.DataLoader(datasets.MNIST(DATA_DIR, train=False,
                    transform=transforms.Compose([transforms.ToTensor()])), batch_size = batch_size, shuffle = True)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        c_dim = 8
        ker_siz = 5
        pad = 2
        self.enc1 = nn.Conv2d(1, c_dim, ker_siz, 2, pad)
        self.enc2 = nn.Conv2d(c_dim, c_dim*2, ker_siz, 2, pad)
        self.enc3 = nn.Conv2d(c_dim*2, c_dim*3, ker_siz, 2, pad)

    def forward(self, x0):
        x1 = F.leaky_relu(self.enc1(x0), 0.1)
        x2 = F.leaky_relu(self.enc2(x1), 0.1)
        x3 = F.leaky_relu(self.enc3(x2), 0.1)
        return x1, x2, x3


class CPPN(nn.Module):

    def __init__(self, w=16, num_layers = 9, activation = nn.Tanh):
        super(CPPN, self).__init__()

        layers = [nn.Linear(3, w), activation()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(w, w), activation()]
        layers += [nn.Linear(w, 1), nn.Sigmoid()]
        self.layers = nn.Sequential(*layers)
        for i in self.layers:
            if type(i) == nn.Linear:
                nn.init.normal_(i.weight)

    def forward(self, x):
        return self.layers(x)


def perceptual_loss(y1, y2, ws = [1.0, 1.0, 1.0]):
    n1, n2, n3 = disc(y1)
    m1, m2, m3 = disc(y2)
    loss1 = ws[0] * F.mse_loss(n1, m1)
    loss2 = ws[1] * F.mse_loss(n2, m2)
    loss3 = ws[2] * F.mse_loss(n3, m3)
    return loss1 + loss2 + loss3

# total variation loss
# https://github.com/jxgu1016/Total_Variation_Loss.pytorch
# https://www.tensorflow.org/tutorials/generative/style_transfer
# https://github.com/tensorflow/tensorflow/blob/v2.7.0/tensorflow/python/ops/image_ops_impl.py#L3213-L3282


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight = 1.0):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


for ii in [1.0, 10.0, 100.0]:

    ws = [1.0,1.0,1.0]

    P1 = ws[0]
    P2 = ws[1]
    P3 = ws[2]
    TVL = ii
    WIDTH = 50
    DEPTH = 10
    model_name = f'cppn_{WIDTH}_{DEPTH}_percep_{P1}_{P2}_{P3}_tvl_{TVL}'

    device      = torch.device('cuda')
    disc        = Discriminator().to(device)
    disc.load_state_dict(torch.load(os.path.join(out_dir, 'mnist_discriminator.pth')), False)

    net         = CPPN(WIDTH, DEPTH).to(device)
    _opt        = torch.optim.Adam(net.parameters(), 0.001)#, weight_decay = 0.00001)

    size        = 28 
    grid        = gen_grid(size).to(device)
    tv_loss     = TVLoss()
    for _e in range(10):
        print(f'epoch {_e}')
        for x, y in train_loader:
            x       = x.to(device) #.reshape([batch_size * 28 * 28, 1])
            y       = y.to(torch.float32) / 4.5 - 1.0
            _labels = torch.repeat_interleave(y[:,None], 784, 0) + torch.randn([784 * batch_size, 1]) * 0.05
            _in     = torch.cat([grid.tile([batch_size,1]), _labels.to(device)], 1)

            _opt.zero_grad()
            _out    = net(_in).reshape([batch_size, 1, 28, 28])
            _loss   = perceptual_loss(_out, x, ws) + TVL * F.mse_loss(_out, x)#+ TVL * tv_loss(_out)
            _loss.backward()
            _opt.step()


    #   B I G   G R I D   1_000
    device = torch.device('cuda')
    #device = torch.device('cpu')
    net.to(device)
    size = 1080 // 2
    grid = gen_grid(size).to(device)
    for i in range(1000):
        _label = torch.Tensor([float(i) / (100.0 * 4.5) - 1.0]).to(device)
        _in = torch.cat([grid, _label[:,None].expand([size*size,1])],1)
        __out = net(_in)
        __result = np.uint8(255.0 * __out.cpu().detach().numpy().squeeze().reshape([size,size]))
        im.fromarray(__result).save(os.path.join(out_dir, f'g0\\_{str(i).zfill(6)}.png'))
        #im.fromarray(__result).show()

    # https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python
    os.chdir(os.path.join(out_dir, 'g0'))
    os.system("ffmpeg -r 60 -i _%06d.png -y " + f'{model_name}.mp4')
    _ = [os.remove(i) for i in os.listdir() if i.split('.')[-1] == 'png']

    torch.save(disc.state_dict(), os.path.join(out_dir, f'{model_name}.pth'))



#   B I G   G R I D   10 
device = torch.device('cpu')
net.to(device)
size = 1080
grid = gen_grid(size).to(device)
for i in range(10):
    _label = torch.Tensor([float(i) / 4.5 - 1.0]).to(device)
    _in = torch.cat([grid, _label[:,None].expand([size*size,1])],1)
    __out = net(_in)
    __result = np.uint8(255.0 * __out.cpu().detach().numpy().squeeze().reshape([size,size]))
    im.fromarray(__result).save(os.path.join(out_dir, f'g0\\_{str(i).zfill(6)}.png'))
    #im.fromarray(__result).show()



'''
https://stackoverflow.com/questions/56187198/scikit-learn-pca-for-image-dataset
---
pca = PCA(n_components=1000, svd_solver='randomized')
pca.fit(X)
Z = pca.transform(X)
---
Also, for images, nonnegative matrix factorisation (NMF) might be better suited. 
For NMF, you can perform stochastic gradient optimisation, subsampling both 
pixels and images for each gradient step.
However, if you still insist on performing PCA, then I think that the 
randomised solver provided by Facebook is the best shot you have. 
Run pip install fbpca and run the following code
from fbpca import pca
# load data into X
U, s, Vh = pca(X, 1000)



https://scicomp.stackexchange.com/questions/36509/how-to-compute-singular-value-decomposition-of-a-large-matrix-with-python
import dask.array as da
x = da.random.random(size=(1_000_000, 20_000), chunks=(20_000, 5_000))
u, s, v = da.linalg.svd_compressed(x, k=10, compute=True)
v.compute()



https://stats.stackexchange.com/questions/41259/how-to-compute-svd-of-a-huge-sparse-matrix
stochastic svd
https://github.com/peisuke/vr_pca * * * 




https://stats.stackexchange.com/questions/7111/how-to-perform-pca-for-data-of-very-high-dimensionality
NIPALS algorithm for performing PCA
https://scikit-learn.org/stable/modules/decomposition.html#pca-using-randomized-svd


https://stats.stackexchange.com/questions/2806/best-pca-algorithm-for-huge-number-of-features-10k/11934#11934


'''


