

import os

IN_KAGGLE = False
if IN_KAGGLE == False:
    DATA_DIR = r'C:\Users\i_hat\Desktop\losable\anime_face_400'
    out_dir = r'C:\Users\i_hat\Desktop\bastl\py\deep_larn\anime_AE'
else:
    DATA_DIR = '/kaggle/input/animefacedataset' # or '../input'
    out_dir = '/kaggle/working'
print(f'length of data set is:\n{len(os.listdir(DATA_DIR))}')
print(f'files in out_dir:\n{len(os.listdir(out_dir))}')

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from torchvision.utils import make_grid, save_image
import tkinter as tk
from tkinter import ttk
import zipfile, pickle
from PIL import Image, ImageTk

os.environ['PYTHONHASHSEED'] = str(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True







class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        # encoder               in, out, kernel_size, stride, padding
        #self.enc1                  = nn.Conv2d(3    , inc  , 5, 2, 2)
        #self.enc2                  = nn.Conv2d(inc  , inc*2, 5, 2, 2)
        #self.enc3                  = nn.Conv2d(inc*2, inc*4, 5, 2, 2)
        #self.enc4                  = nn.Conv2d(inc*4, inc*8, 5, 2, 2)
        #self.enc5                  = nn.Conv2d(inc*8, 1024 , 5, 2, 2)
        #self.flat                  = nn.Flatten()
        #self.enc_fc                = nn.Linear(1024 * 2 * 2, latent_dim)
        self.enc_fc                = nn.Linear(64 * 64 * 3, latent_dim)

        # decoder 
        self.dec_fc                = nn.Linear(latent_dim, 1024 * 2 * 2)
        self.dec1                  = nn.ConvTranspose2d(inc*16, inc*8, 2, 2)
        self.dec2                  = nn.ConvTranspose2d(inc*8, inc*4, 2, 2)
        self.dec3                  = nn.ConvTranspose2d(inc*4, inc*2, 2, 2)
        self.dec4                  = nn.ConvTranspose2d(inc*2, inc, 2, 2)
        self.dec5                  = nn.ConvTranspose2d(inc, 32, 2, 2)
        self.dec6                  = nn.Conv2d(32, 3, 5, 1, 2)

    def encode(self, x):
        #x                          = F.relu(self.enc1(x))                     # [batch_size, 64, 32, 32] data shape is [batch_size, 3, 64, 64]
        #x                          = F.relu(self.enc2(x))                     # [batch_size, 128, 16, 16]
        #x                          = F.relu(self.enc3(x))                     # [batch_size, 256, 8, 8]
        #x                          = F.relu(self.enc4(x))                     # [batch_size, 512, 4, 4]
        #x                          = F.relu(self.enc5(x))                     # [batch_size, 1024, 2, 2]
        #enc                        = F.relu(self.enc_fc(self.flat(x)))        # [batch_size, 100]
        #return enc
        return torch.tanh(self.enc_fc(x))

    def decode(self, z):
        z                          = F.relu(self.dec_fc(z))                 # [batch_size, 4096]
        z                          = z.reshape([-1, 1024, 2, 2])              # [batch_size, 1024, 2, 2]
        z                          = F.relu(self.dec1(z))                     # [batch_size, 512, 4, 4]
        z                          = F.relu(self.dec2(z))                     # [batch_size, 256, 8, 8]
        z                          = F.relu(self.dec3(z))                     # [batch_size, 128, 16, 16]
        z                          = F.relu(self.dec4(z))                     # [batch_size, 64, 32, 32]
        z                          = F.relu(self.dec5(z))                     # [batch_size, 32, 64, 64]
        recon                      = torch.sigmoid(self.dec6(z))              # [batch_size, 3, 64, 64]
        return recon
 
    def forward(self, x):
        enc = self.encode(x)
        return self.decode(enc), enc



latent_dim                         = 20 # number of features to consider
batch_size                         = 64
inc                                = 64  # initial number of filters
x_train = torchvision.datasets.ImageFolder(root = DATA_DIR, transform=transforms.Compose([
                                      transforms.Resize(64),
                                      transforms.CenterCrop(64),
                                      transforms.ToTensor()]))
trainloader                         = torch.utils.data.DataLoader(x_train, batch_size, drop_last=True, shuffle=True)
device                              = torch.device('cpu') # torch.cuda.is_available()
ae                                  = AE()
#ae.load_state_dict(torch.load(os.path.join(out_dir, r'k2\ae_20.pth')))
_ = ae.to(device)
optimizer                           = torch.optim.Adam(ae.parameters(), 0.001) #  weight_decay = 0.1




_ = ae.train()
losses                 = []
for _e in range(100):
    print(f'starting {_e}')
    for data, _ in trainloader:
        optimizer.zero_grad()
        data                       = data.to(device)
        #recons, encoded            = ae(data)
        recons, encoded            = ae(data.view(-1, 64*64*3))
        loss                       = F.mse_loss(recons, data)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    save_image(recons.cpu(), os.path.join(out_dir, f"output{_e}.jpg"))
    print('epoch %d, loss %.4f' % (_e, losses[-1]))
    if _e % 5 == 0:
        torch.save(ae.state_dict(), os.path.join(out_dir, f"ae_{_e}.pth"))








# save all output imgs
f = [i for i in os.listdir(out_dir) if i.split('.')[-1] in ['jpg', 'pth', 'gif']]
with zipfile.ZipFile(os.path.join(out_dir, '_results.zip'),'w' ) as myzip:
    for i in f:
        myzip.write(i)
print('done')






ae.eval()

# get encoding of data set
codes = np.zeros([len(trainloader) * batch_size, 50], dtype = np.float32)
for c, (batch, _) in enumerate(trainloader):
    #codes[c*batch_size:(c+1)*batch_size,:] = ae.encode(batch.view(-1, 3 * 64 * 64)).detach().numpy()
    codes[c*batch_size:(c+1)*batch_size,:] = ae.encode(batch).detach().numpy()


# do PCA on encoding
def eig(S):
    va, vc          = np.linalg.eigh(S)  
    _sorted         = np.argsort(-va) # sorting them in decrasing order
    va              = va[_sorted]
    vc              = vc[:, _sorted]
    return (va, vc)
mu                  = np.mean(codes, 0, keepdims = True)
_codes              = codes - mu
_std                = np.std(_codes, 0, keepdims = True)
_std[_std == 0.0]   = 1.0
c                   = _codes / _std
my_cov              = (c.T @ c) / (c.shape[0] - 1)
vals, vecs          = eig(my_cov) 


siz = 64


# do pca
components          = 15
U                   = vecs[:, range(components)]       # take a subset


def from_latent(z, add_mean = True):
    x = z @ U.T
    if add_mean:
        return x * _std + mu
    return x * _std




class Viewer():
    def reconstruct(self):
        code = np.array([[i.get() for i in self.vals]])
        self.x = 255.0 * ae.decode(torch.Tensor(from_latent(code))).squeeze().permute([1,2,0]).cpu().detach().numpy()
        self.x = np.clip(self.x, 0.0, 255.0).astype(np.uint8)

    def refresh(self, e):
        self.reconstruct()
        self.photo = ImageTk.PhotoImage(image = Image.fromarray(self.x)) # https://stackoverflow.com/questions/58411250/photoimage-zoom
        self.photo = self.photo._PhotoImage__photo.zoom(8)
        self.canvas_area.create_image(0,0,image = self.photo, anchor=tk.NW)
        self.canvas_area.update()

    def key_press(self, e):
        if e.char                                        == 'r':      # reset sliders to 0
            [self.vals[i].set(0.0) for i in range(len(self.vals))]
        elif e.char                                      == 'a':      # toggle mean option 
            self.add_mean                                = not self.add_mean
        elif e.char                                      == 't':      # randomize
            num_feats                                    = min(self.num_sliders, 50) # dont do all the features as most are noise
            rand                                         = np.clip(np.random.randn(num_feats) * 2.0, -50.0, 50.0)
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

    def start(self, num_sliders                          = 100):
        self.root                                        = tk.Tk()
        self.menu_left                                   = tk.Canvas(self.root, width=150, height = 400, bg = 'black')
        self.menu_left.grid(row                          = 0, column=0, sticky = 'nsew')
        sf                                               = ttk.Frame(self.menu_left)
        sf.bind("<Configure>",  lambda e: self.menu_left.configure(scrollregion = self.menu_left.bbox("all")))
        self.root.bind('<Key>', lambda x: self.key_press(x))
        self.menu_left.create_window((0, 0), window      =sf, anchor="nw")

        self.remember_vecs                               = []
        self.num_sliders                                 = num_sliders
        self.add_mean                                    = True
        self.vals                                        = [tk.DoubleVar() for i in range(self.num_sliders)]
        labs                                             = [ttk.Label(sf, text=f"{i}") for i in range(self.num_sliders)]
        slds                                             = [None for i in range(self.num_sliders)]
        for i in range(self.num_sliders):
            slds[i]                                      = ttk.Scale(sf, from_ = -5.0, to = 5.0, orient = 'horizontal', variable = self.vals[i], command = self.refresh)
            slds[i].grid(column                          = 1, row = i, columnspan = 1, sticky = 'nsew')
            labs[i].grid(column                          = 0, row = i, columnspan = 1, sticky = 'nsew')

        self.canvas_area                                 = tk.Canvas(self.root, width=540, height=540, bg = 'black')
        self.canvas_area.grid(row                        = 0, column=1, sticky = 'nsew') 
        self.root.grid_rowconfigure(1, weight            = 1)
        self.root.grid_columnconfigure(1, weight         = 1)
        self.refresh(0)
        self.root.mainloop()
        

v                                                        = Viewer()
v.start(components)


''' AE NOTES
- being nonlinear is very cool but it is way too chaotic. the features are all entangled. each slider changes 10 things. 
- i think that may be a big problem with this: that the latent space is too big so it encodes all the noise. 
- interestingly, the reconstructions of the embedding-only AE (hacker poet) are a lot cripser. i think this is because 
    the model didn't capture any average behavior and just mapped the points to latent space with a matrix while the decoder overfit. 
    using a conv arch gives somewhat blurry recons (also latent dim was 50)

> try smaller latent space
> what if i use weight decay?


- tried weight decay of 0.1, images all grey. 


* * * TRYING NOW IN KAGGLE:
- use tanh after linear embedding. this is like a poor man's VAE, to squish the embeddings together. 
- if this does anything remotely good, try with smol weight decay. 

'''

