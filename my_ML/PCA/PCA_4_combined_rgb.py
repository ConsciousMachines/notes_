
import numpy as np
import tkinter as tk
from tkinter import ttk
import os, pickle
from PIL import Image, ImageTk


siz                 = 64     # picture length and width
SAVE_DIR            = r'C:\Users\i_hat\Desktop\bastl\py\deep_larn\anime_PCA'
IMG_DIR             = r'C:\Users\i_hat\Desktop\losable\anime_face_400\images'


def generate_chunk_inds(total, sample_size): # generate indices for chunks of data 
    _inds = list(range(0, total, sample_size)) + [total]
    inds = [(_inds[i], _inds[i+1]) for i in range(len(_inds)-1)]
    return inds


def get_data_chunk(_start, _end, siz = siz, IMG_DIR = IMG_DIR): # retrieve data between a range of indices
    _samples              = _end - _start
    x_orig_r              = np.zeros([_samples, siz * siz])
    x_orig_g              = np.zeros([_samples, siz * siz])
    x_orig_b              = np.zeros([_samples, siz * siz])
    for i, img in enumerate(os.listdir(IMG_DIR)[_start: _end]):
        _img              = Image.open(os.path.join(IMG_DIR, img))
        assert _img.mode  == 'RGB'
        _data             = np.array(_img.resize((siz, siz), Image.ANTIALIAS))
        x_orig_r[i,:]     = _data[:,:,0].flatten()
        x_orig_g[i,:]     = _data[:,:,1].flatten()
        x_orig_b[i,:]     = _data[:,:,2].flatten()
    return x_orig_r, x_orig_g, x_orig_b


def get_chunk_cov(x, mu, std, n):
    _x = (x - mu) / std
    return _x.T @ _x / n


def eig(S):
    va, vc          = np.linalg.eigh(S)  
    _sorted         = np.argsort(-va) # sorting them in decrasing order
    va              = va[_sorted]
    vc              = vc[:, _sorted]
    return (va, vc)


if False:

    # we already gathered the means and the stds before. 
    vecsr, vecsg, vecsb, mu_r, mu_g, mu_b, std_r, std_g, std_b = pickle.load(open(os.path.join(SAVE_DIR, 'anime_400_PCA_3.pkl'), 'rb'))
    mu                                                         = np.concatenate([mu_r, mu_g, mu_b], 1)
    std                                                        = np.concatenate([std_r, std_g, std_b], 1)

    # S T E P   3   :   C O V A R I A N C E 
    inds           = generate_chunk_inds(len(os.listdir(IMG_DIR)), 10_000)
    my_cov         = np.zeros([siz*siz*3, siz*siz*3])
    for start, end in inds:
        data_chunk = get_data_chunk(start, end)
        x          = np.concatenate([data_chunk[0], data_chunk[1], data_chunk[2]], 1)
        my_cov    += get_chunk_cov(x, mu, std, inds[-1][-1])

    # S T E P   4   :   P C A
    _, vecs = eig(my_cov)
    #pickle.dump([mu, std, vecs], open(os.path.join(SAVE_DIR, 'anime_400_PCA_4.pkl'), 'wb'))
    #pickle.dump(my_cov, open(os.path.join(SAVE_DIR, 'anime_400_PCA_4_my_cov.pkl'), 'wb'))

    #==================== start experimetn - apparently i did it correct this time. 
    #my_cov = pickle.load(open(os.path.join(SAVE_DIR, 'anime_400_PCA_4_my_cov.pkl'), 'rb'))
    #m1 = np.ones([siz*siz, siz*siz])
    #m0 = np.zeros([siz*siz, siz*siz])
    #my_cov = my_cov * np.concatenate([
    #    np.concatenate([m1,m0,m0]),
    #    np.concatenate([m0,m1,m0]),
    #    np.concatenate([m0,m0,m1]),
    #], 1)
    #_, vecs = eig(my_cov)
    #==================== end experimetn



siz, components, mean, std, U = pickle.load(open(os.path.join('/home/chad/Desktop/_backups/py/my_ML/PCA/pkl', 'PCA_5_anime_400_64.pkl'), 'rb'))



def from_latent(z, add_mean = True): 
    x = z @ U.T # comes back from latent space 
    if add_mean:
        return x * std + mean # when we transpose from the latent space, we need to unstandardize it
    return x * std


class Viewer():
    def reconstruct(self):
        code = np.array([[i.get() for i in self.vals]])
        self.x = np.clip(from_latent(code, self.add_mean),0,255).astype(np.uint8).reshape([3,siz,siz]).transpose([1,2,0])

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
            num_feats                                    = min(self.num_sliders, 40) # dont do all the features as most are noise
            rand                                         = np.clip(np.random.randn(num_feats) * 6.0, -50.0, 50.0)
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
            slds[i]                                      = ttk.Scale(sf, from_ = -50.0, to = 50.0, orient = 'horizontal', variable = self.vals[i], command = self.refresh)
            slds[i].grid(column                          = 1, row = i, columnspan = 1, sticky = 'nsew')
            labs[i].grid(column                          = 0, row = i, columnspan = 1, sticky = 'nsew')

        self.canvas_area                                 = tk.Canvas(self.root, width=540, height=540, bg = 'black')
        self.canvas_area.grid(row                        = 0, column=1, sticky = 'nsew') 
        self.root.grid_rowconfigure(1, weight            = 1)
        self.root.grid_columnconfigure(1, weight         = 1)
        self.refresh(0)
        self.root.mainloop()


v                                                 = Viewer()
v.start(components)






import matplotlib.pyplot as plt
from matplotlib import animation


steps = 30
frames = np.zeros([steps * len(v.remember_vecs), components])

for i in range(len(v.remember_vecs)):
    v1 = v.remember_vecs[i]
    v2 = v.remember_vecs[(i+1) % len(v.remember_vecs)]
    dv = v2 - v1
    for j in range(steps):
        frames[i*steps + j,:] = v1 + dv * (float(j) / steps)

fig = plt.figure()
_ = plt.axis('off')
images = []
for c in range(frames.shape[0]):
    morphed = np.clip(from_latent(frames[[c],:]),0,255).astype(np.uint8).reshape([3,siz,siz]).transpose([1,2,0])
    images.append([plt.imshow(morphed, animated=True)])

ani = animation.ArtistAnimation(fig, images, interval=30)
ani.save(os.path.join(SAVE_DIR, 'ani.mp4'), writer = animation.FFMpegWriter(fps = 60))
ani.save(os.path.join(SAVE_DIR, 'ani.mp4'), writer = animation.FFMpegWriter(fps = 60))

