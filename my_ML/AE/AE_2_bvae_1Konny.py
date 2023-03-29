# https://github.com/1Konny/Beta-VAE

import os, zipfile
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import save_image

IN_KAGGLE = False
if IN_KAGGLE:
    DATA_DIR = '/kaggle/input/celeba-dataset/img_align_celeba' # or '../input'
    DATA_DIR = '/kaggle/input/animefacedataset' # or '../input'
    out_dir = '/kaggle/working'
else:
    out_dir = r'C:\Users\i_hat\Desktop\bastl\py\deep_larn\beta_VAE\1Konny'
    DATA_DIR = r'C:\Users\i_hat\Desktop\losable\celeba\img_align_celeba'
    DATA_DIR = r'C:\Users\i_hat\Desktop\losable\anime_face_400'


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))
    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise_kld, mean_kld


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),  # B, nc, 64, 64
        )
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self.encoder(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = torch.sigmoid(self.decoder(z))
        return x_recon, mu, logvar


class BetaVAE_B(BetaVAE_H):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=10):
        super(BetaVAE_B, self).__init__()
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 32*4*4)),                  # B, 512
            nn.Linear(32*4*4, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), # B,  nc, 64, 64
        )
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self.encoder(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = torch.sigmoid(self.decoder(z).view(x.size()))
        return x_recon, mu, logvar


batch_size = 128
z_dim = 40           
beta = 4                
#gamma = 1000            
#C_max = 25              
#C_stop_iter = 1e5       
device = torch.device('cuda')

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

net = BetaVAE_B(z_dim).to(device)



optim = optim.Adam(net.parameters(), lr = 1e-3)
transform = transforms.Compose([
    #transforms.CenterCrop((178,178)),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),])
train_data = ImageFolder(root = DATA_DIR, transform = transform)
data_loader = DataLoader(train_data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2,
                            pin_memory=True,
                            drop_last=True)

#C_max = Variable(torch.FloatTensor([C_max]))

for _e in range(100):
    print(f'epoch {_e}')
    for x, _ in data_loader:
        x = x.to(device)

        x_recon, mu, logvar = net(x)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum').div(x.size(0))
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

        beta_vae_loss = recon_loss + beta*total_kld
        #C = torch.clamp(C_max/ C_stop_iter*_i, 0, C_max.data[0])
        #beta_vae_loss = recon_loss + gamma*(total_kld-C).abs()

        optim.zero_grad()
        beta_vae_loss.backward()
        optim.step()

    if _e % 5 == 0:
        torch.save(net.state_dict(), os.path.join(out_dir, f"ae_{_e}.pth"))
    save_image(x_recon.cpu(), os.path.join(out_dir, f"output{_e}.jpg"))



# save all output imgs
f = [i for i in os.listdir(out_dir) if i.split('.')[-1] in ['jpg', 'pth', 'gif']]
with zipfile.ZipFile(os.path.join(out_dir, '_results.zip'),'w' ) as myzip:
    for i in f:
        myzip.write(i)
print('done')







net.load_state_dict(torch.load(os.path.join(out_dir, '2\\ae_95.pth')))
device = torch.device('cpu')
_ = net.eval().to(device)



# I GOT WEIRD COLOR BC NET.DECODER DOESNT INCLUDE SIGMOID

import numpy as np
import tkinter as tk
from tkinter import ttk
import zipfile, io, os, pickle
from PIL import Image, ImageTk

siz = 64

class Viewer():
    def reconstruct(self):
        code = torch.Tensor([[i.get() for i in self.vals]])
        #self.x = np.clip(from_latent(code, self.add_mean),0,255).astype(np.uint8)
        self.x = torch.clip(255.0 * torch.sigmoid(net.decoder(code)).squeeze().permute([1,2,0]), 0.0, 255.0).to(torch.uint8).numpy()

    def refresh(self, e):
        self.reconstruct()
        self.photo = ImageTk.PhotoImage(image = Image.fromarray(self.x)) # https://stackoverflow.com/questions/58411250/photoimage-zoom
        self.photo = self.photo._PhotoImage__photo.zoom(self.zoom)
        self.canvas_area.create_image(0,0,image = self.photo, anchor=tk.NW)
        self.canvas_area.update()

    def key_press(self, e):
        if e.char                                        == 'r':      # reset sliders to 0
            [self.vals[i].set(0.0) for i in range(len(self.vals))]
        elif e.char                                      == 'a':      # toggle mean option 
            self.add_mean                                = not self.add_mean
        elif e.char                                      == 't':      # randomize
            num_feats                                    = min(self.num_sliders, 40) # dont do all the features as most are noise
            rand                                         = np.clip(np.random.randn(num_feats) * self.std, -self.slider_max, self.slider_max)
            for i in range(num_feats):
                self.vals[i].set(rand[i])
        elif e.char                                      == 'e':      # remember the encoding 
            self.remember_vecs.append(np.array([[i.get() for i in self.vals]]))
        elif e.keysym                                    == 'Escape': # quit 
            return self.root.destroy()
        elif e.keysym                                    == 'Down':   # scroll down in slider menu
            self.menu_left.yview_scroll(10, "units")
        elif e.keysym                                    == 'Up':     # scroll up in slider menu
            self.menu_left.yview_scroll(-10, "units")
        else:
            print(e)
        self.refresh(0)

    def start(self, num_sliders                          = 100, siz = 128, slider_max = 100.0, std = 8.0):
        self.root                                        = tk.Tk()
        self.menu_left                                   = tk.Canvas(self.root, width=150, height = 400, bg = 'black')
        self.menu_left.grid(row                          = 0, column=0, sticky = 'nsew')
        sf                                               = ttk.Frame(self.menu_left)
        sf.bind("<Configure>",  lambda e: self.menu_left.configure(scrollregion = self.menu_left.bbox("all")))
        self.root.bind('<Key>', lambda x: self.key_press(x))
        self.menu_left.create_window((0, 0), window      =sf, anchor="nw")

        self.remember_vecs                               = []
        self.std                                         = std
        self.slider_max                                  = slider_max
        self.siz                                         = siz
        self.zoom                                        = 512 // siz
        self.num_sliders                                 = num_sliders
        self.add_mean                                    = True
        self.vals                                        = [tk.DoubleVar() for i in range(self.num_sliders)]
        labs                                             = [ttk.Label(sf, text=f"{i}") for i in range(self.num_sliders)]
        slds                                             = [None for i in range(self.num_sliders)]
        for i in range(self.num_sliders):
            slds[i]                                      = ttk.Scale(sf, from_ = -slider_max, to = slider_max, orient = 'horizontal', variable = self.vals[i], command = self.refresh)
            slds[i].grid(column                          = 1, row = i, columnspan = 1, sticky = 'nsew')
            labs[i].grid(column                          = 0, row = i, columnspan = 1, sticky = 'nsew')

        self.canvas_area                                 = tk.Canvas(self.root, width=540, height=540, bg = 'black')
        self.canvas_area.grid(row                        = 0, column=1, sticky = 'nsew') 
        self.root.grid_rowconfigure(1, weight            = 1)
        self.root.grid_columnconfigure(1, weight         = 1)
        self.refresh(0)
        self.root.mainloop()


v                                                 = Viewer()
v.start(z_dim, siz, slider_max = 10.0, std = 2.0)

