

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation

out_dir = r'C:\Users\i_hat\Desktop\bastl\py\deep_larn'
DATA_DIR = r'C:\Users\i_hat\Desktop\losable\pytorch_data'
batch_size = 128

train_loader = torch.utils.data.DataLoader(datasets.MNIST(DATA_DIR, train=True,
                    transform=transforms.Compose([transforms.ToTensor()])), batch_size = batch_size, shuffle=True, drop_last = True)

test_loader = torch.utils.data.DataLoader(datasets.MNIST(DATA_DIR, train=False,
                    transform=transforms.Compose([transforms.ToTensor()])), batch_size = batch_size, shuffle = True)



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        c_dim = 8
        h_dim = 64
        ker_siz = 5
        pad = 2
        self.enc1 = nn.Conv2d(1, c_dim, ker_siz, 2, pad)
        self.enc2 = nn.Conv2d(c_dim, c_dim*2, ker_siz, 2, pad)
        self.enc3 = nn.Conv2d(c_dim*2, c_dim*3, ker_siz, 2, pad)
        self.fc1  = nn.Linear(4 * 4 * 3 * c_dim, h_dim)
        self.fc2  = nn.Linear(h_dim, 10)

    def forward(self, x):
        x = F.leaky_relu(self.enc1(x), 0.1)
        x = F.leaky_relu(self.enc2(x), 0.1)
        x = F.leaky_relu(self.enc3(x), 0.1)
        x = x.flatten(1)
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = self.fc2(x) # removing sigmoid makes me go from 9 to 96% acc in one epoch

        #x = F.relu(self.enc1(x))
        #x = F.relu(self.enc2(x))
        #x = F.relu(self.enc3(x))
        #x = x.flatten(1)
        #x = F.relu(self.fc1(x))
        #x = self.fc2(x) # removing sigmoid makes me go from 9 to 96% acc in one epoch
        return x


device = torch.device('cuda')
device = torch.device('cpu')
disc = Discriminator()
disc.load_state_dict(torch.load(os.path.join(out_dir, 'mnist_discriminator.pth')))
_ = disc.to(device)
opt = torch.optim.AdamW(disc.parameters(), 0.001)

_ = disc.train()
for _e in range(10):
    # test accuracy 
    _ = disc.eval()
    acc = 0
    for data, y in test_loader:
        data = data.to(device)
        y = y.to(device)
        acc += torch.sum(torch.argmax(disc(data), 1) == y)
    print(f'epoch {_e} acc {acc / 10_000}')
    if acc / 10_000 > 0.99:
        break
    # train
    _ = disc.train()
    for data, y in train_loader:
        opt.zero_grad()
        data = data.to(device)
        _loss = F.mse_loss(disc(data), F.one_hot(y, 10).to(device, torch.float32))
        _loss.backward()
        opt.step()


#torch.save(disc.state_dict(), os.path.join(out_dir, 'mnist_discriminator.pth'))


device = torch.device('cpu')
_ = disc.to(device)
x0 = data[[0],:,:,:].to(device)
x1 = disc.enc1(x0)
x2 = disc.enc2(x1)
x3 = disc.enc3(x2)
x3_ = x3.flatten(1)
x4 = disc.fc1(x3_)
x5 = disc.fc2(x4)



fig, ax = plt.subplots(1, 6, figsize=(9, 3))
_ = ax[0].imshow(x0.reshape([28,28]))
_ = ax[0].axis('off')
_ = ax[0].set_title('orig')

_x1 = x1.cpu().detach().numpy()
_x1.shape
x_dim_siz = 4
y_dim_siz = 2
pic = np.zeros([x_dim_siz*14,y_dim_siz*14])
for i in range(x_dim_siz):
    for j in range(y_dim_siz):
        pic[i*14:(i+1)*14,j*14:(j+1)*14] = _x1[0,i*y_dim_siz+j,:,:]
_ = ax[1].imshow(pic)
_ = ax[1].axis('off')
_ = ax[1].set_title('conv1')

_x2 = x2.cpu().detach().numpy()
_x2.shape
x_dim_siz = 4
y_dim_siz = 4
pic = np.zeros([x_dim_siz*7,y_dim_siz*7])
for i in range(x_dim_siz):
    for j in range(y_dim_siz):
        pic[i*7:(i+1)*7,j*7:(j+1)*7] = _x2[0,i*y_dim_siz+j,:,:] # i think its super light bc of weight decay
_ = ax[2].imshow(pic)
_ = ax[2].axis('off')
_ = ax[2].set_title('conv2')


_x3 = x3.cpu().detach().numpy()
_x3.shape
x_dim_siz = 6
y_dim_siz = 4
pic = np.zeros([x_dim_siz * 4, y_dim_siz * 4])
for i in range(x_dim_siz):
    for j in range(y_dim_siz):
        pic[i*4:(i+1)*4,j*4:(j+1)*4] = _x3[0,i*y_dim_siz+j,:,:]
_ = ax[3].imshow(pic)
_ = ax[3].axis('off')
_ = ax[3].set_title('conv3')


_x4 = x4.cpu().detach().numpy()
_x4.shape
_ = ax[4].imshow(_x4.reshape([16, 4]))
_ = ax[4].axis('off')
_ = ax[4].set_title('dense1')


_x5 = x5.cpu().detach().numpy().T
_ = ax[5].imshow(_x5)
_ = ax[5].set_xticks([])
_ = ax[5].set_yticks(list(range(10)))
_ = ax[5].set_title('dense2')

plt.show()






# animation 
# https://brushingupscience.com/2016/06/21/matplotlib-animations-the-easy-way/

if False:
        
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    x = np.linspace(-3, 3, 91)
    t = np.linspace(0, 25, 30)
    y = np.linspace(-3, 3, 91)
    X3, Y3, T3 = np.meshgrid(x, y, t)
    sinT3 = np.sin(2*np.pi*T3 /
                T3.max(axis=2)[..., np.newaxis])
    G = (X3**2 + Y3**2)*sinT3


    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set(xlim=(-3, 3), ylim=(-1, 1))

    cax = ax.pcolormesh(x, y, G[:-1, :-1, 0],
                        vmin=-1, vmax=1, cmap='Blues')
    fig.colorbar(cax)
    
    def animate(i):
        cax.set_array(G[:-1, :-1, i].flatten())

    anim = FuncAnimation(
        fig, animate, interval=100, frames=len(t)-1)
    
    plt.draw()
    plt.show()











# add last entry again so we loop 
my_data = torch.cat([data, data[[0],:,:,:]]).reshape(-1)


fig, ax = plt.subplots(1, 6, figsize=(9, 3))
_ = ax[0].axis('off')
_ = ax[0].set_title('orig')
_ = ax[1].axis('off')
_ = ax[1].set_title('conv1')
_ = ax[2].axis('off')
_ = ax[2].set_title('conv2')
_ = ax[3].axis('off')
_ = ax[3].set_title('conv3')
_ = ax[4].axis('off')
_ = ax[4].set_title('dense1')
_ = ax[5].set_xticks([])
_ = ax[5].set_yticks(list(range(10)))
_ = ax[5].set_title('dense2')
_x1 = x1.cpu().detach().numpy()
_x2 = x2.cpu().detach().numpy()
_x3 = x3.cpu().detach().numpy()
_x4 = x4.cpu().detach().numpy()
_x5 = x5.cpu().detach().numpy().T

ax0 = ax[0].imshow(x0.reshape([28,28]))

x_dim_siz = 4
y_dim_siz = 2
pic = np.zeros([x_dim_siz*14,y_dim_siz*14])
for i in range(x_dim_siz):
    for j in range(y_dim_siz):
        pic[i*14:(i+1)*14,j*14:(j+1)*14] = _x1[0,i*y_dim_siz+j,:,:]
ax1 = ax[1].imshow(pic)

x_dim_siz = 4
y_dim_siz = 4
pic = np.zeros([x_dim_siz*7,y_dim_siz*7])
for i in range(x_dim_siz):
    for j in range(y_dim_siz):
        pic[i*7:(i+1)*7,j*7:(j+1)*7] = _x2[0,i*y_dim_siz+j,:,:]
ax2 = ax[2].imshow(pic)


x_dim_siz = 6
y_dim_siz = 4
pic = np.zeros([x_dim_siz * 4, y_dim_siz * 4])
for i in range(x_dim_siz):
    for j in range(y_dim_siz):
        pic[i*4:(i+1)*4,j*4:(j+1)*4] = _x3[0,i*y_dim_siz+j,:,:]
ax3 = ax[3].imshow(pic)


ax4 = ax[4].imshow(_x4.reshape([16, 4]))
ax5 = ax[5].imshow(_x5)




def animate(i):
    global my_data
    #x0 = data[[i],:,:,:].to(device)
    x0 = my_data[i*28:i*28 + 784].reshape(1,1,28,28).to(device)

    x1 = F.leaky_relu(disc.enc1(x0), 0.1)
    x2 = F.leaky_relu(disc.enc2(x1), 0.1)
    x3 = F.leaky_relu(disc.enc3(x2), 0.1)
    x3_ = x3.flatten(1)
    x4 = F.leaky_relu(disc.fc1(x3_), 0.1)
    x5 = disc.fc2(x4)
    _x1 = x1.cpu().detach().numpy()
    _x2 = x2.cpu().detach().numpy()
    _x3 = x3.cpu().detach().numpy()
    _x4 = x4.cpu().detach().numpy()
    _x5 = x5.cpu().detach().numpy().T

    ax0.set_array(x0.reshape([28,28]))

    x_dim_siz = 4
    y_dim_siz = 2
    pic = np.zeros([x_dim_siz*14,y_dim_siz*14])
    for i in range(x_dim_siz):
        for j in range(y_dim_siz):
            pic[i*14:(i+1)*14,j*14:(j+1)*14] = _x1[0,i*y_dim_siz+j,:,:]
    #ax1.set_array(pic)
    ax[1].imshow(pic)

    x_dim_siz = 4
    y_dim_siz = 4
    pic = np.zeros([x_dim_siz*7,y_dim_siz*7])
    for i in range(x_dim_siz):
        for j in range(y_dim_siz):
            pic[i*7:(i+1)*7,j*7:(j+1)*7] = _x2[0,i*y_dim_siz+j,:,:]
    #ax2.set_array(pic)
    ax[2].imshow(pic)

    x_dim_siz = 6
    y_dim_siz = 4
    pic = np.zeros([x_dim_siz * 4, y_dim_siz * 4])
    for i in range(x_dim_siz):
        for j in range(y_dim_siz):
            pic[i*4:(i+1)*4,j*4:(j+1)*4] = _x3[0,i*y_dim_siz+j,:,:]
    #ax3.set_array(pic)
    ax[3].imshow(pic)

    #ax4.set_array(_x4.reshape([16, 4]))
    ax[4].imshow(_x4.reshape([16, 4]))
    #ax5.set_array(_x5)
    ax[5].imshow(_x5)


anim = FuncAnimation(fig, animate, interval=100, frames = 28 * 16)
#plt.draw()
#plt.show()
anim.save(os.path.join(out_dir, 'cnn_viz_mnist.mp4'))#, writer = animation.FFMpegWriter(fps=60))

