'''

import cupy as cp
cp.cuda.Device()

x_cpu = np.array([1, 2, 3])
x_gpu = cp.asarray(x_cpu)  # move the data to the current device.

with cp.cuda.Device(0):
    x_gpu_0 = cp.ndarray([1, 2, 3])  # create an array in GPU 0
x_cpu = cp.asnumpy(x_gpu)  # move the array to the host.
'''


# MODIFIED FOR n-1


# i think that if we have n samples and m features, and n is large, our cov matrix is m x m, 
# and the process can be split up into chunks where we take b = n / 10 at a time.
# for example, the first entry is, for i in [1..n]:
# var(x1) = 1/(n-1) sum_1_n (x1i - x1bar)^2
# similarly, 
# cov(x1, x2) = 1/(n-1) sum_1_n (x1i - x1bar) * (x2i - x2bar)
# so it is a single summation, which means we can partition n into independent parts. 
# say we split n into 2 parts, [n1,n2]. 
# say n1 = n//2, n2 = n
# we first get the means of [n1,n2] and then get the full mean by taking the mean of means.
# then we can compute:
# sum__1_n1 (x1i - x1bar) * (x2i - x2bar)
# sum_n1_n2 (x1i - x1bar) * (x2i - x2bar)
# then add these and divide by (n-1) to get cov(x1, x2). 

# we need to do a first pass to get the means.
# x1bar = 1 / n sum_1_n x1i 
# we will probably have one batch with a weird size so we can rewrite:
# x1bar = 1 / n ( sum_1_n1 x1i + sum_n1_n2 x1i )
# x1bar = 1 / n ( (n1-0) * mean_x1(1,n1) + (n2-n1) * mean_x1(n1,n2))
# x1bar = (n1-0) / n * mean_x1(1,n1) + (n2-n1) / n * mean_x1(n1,n2)
# so we can get the means of each batch and remember their weights. 

# the second step is to go over the data AGAIN and calculate its std 
# var = 1/(n-1) sum_1_n (x1i - x1bar)^2
# std = SQRT var
# we can compute the var of each chunk.
# var = 1/(n-1) ( sum_1_n1 (x1i - x1bar)^2  +  sum_n1_n2 (x1i - x1bar)^2)
# mse(n1,n2,x1bar) = 1/(n2-n1) * sum_n1_n2 (x1i - x1bar)^2
# var = (n1-0)/(n-1) * mse(1,n1,x1bar) + (n2-n1)/(n-1) * mse(n1,n2,x1bar)

# the fact that we need 3 steps to go over the data can be seen by this function:
#def standardize(inp):
#    mu                  = np.mean(inp, axis = 0, keepdims = True)
#    std                 = np.std(inp - mu, axis = 0, keepdims = True)# calculate variance of features
#    x                   = (inp - mu) / std                             # scale the data
#    return x, mu, std
# mu depends on all x so that's one step. std depends on all x and mu, that's a second step.
# x depends on input, mu, and std so that is a third step. 





import zipfile, io, pickle, os
import numpy as np
import PIL.Image as im

components     = 50
siz            = 128
SAVE_DIR       = r'C:\Users\i_hat\Desktop\bastl\py\deep_larn\anime_PCA\p'

#DATA_DIR       = r'C:\Users\i_hat\Desktop\losable\anime_face_400\images'
#DATA_DIR       = r'C:\Users\i_hat\Desktop\losable\anime_aligned.zip'
#DATA_DIR        = r'C:\Users\i_hat\Desktop\losable\anime_scrib\animefaces256cleaner'
DATA_DIR        = r'C:\Users\i_hat\Desktop\losable\celeba\img_align_celeba\img_align_celeba'


def generate_chunk_inds(total, sample_size): # generate indices for chunks of data 
    _inds = list(range(0, total, sample_size)) + [total]
    inds = [(i, _inds[i], _inds[i+1]) for i in range(len(_inds)-1)]
    return inds


def get_sample_weights(inds, n): # get the weights of the samples: ni / n  
    return [(end - start) / n for _, start, end in inds]


def get_data_chunk(start, end, files, siz): 
    # if file was cached, return it
    file_name = os.path.join(SAVE_DIR, f'cache_x_{start}_{end}.pkl')
    if os.path.exists(file_name):
        print('file found in cache')
        return pickle.load(open(file_name, 'rb'))
    
    x                  = np.zeros([end - start, siz * siz * 3], dtype = np.uint8) 
    for i in range(start, end):
        orig_im        = im.open(files[i]) # im.open(io.BytesIO(zip.open(files[i]).read()))
        crop_im = orig_im.crop((22,64,156,198)).resize([siz,siz]) 
        x[i - start,:] = np.array(crop_im, dtype = np.uint8).flatten()

    # cache the file
    pickle.dump(x, open(file_name,'wb'))
    return x


#zip            = zipfile.ZipFile(DATA_DIR)
#files          = zip.namelist()
files = [os.path.join(DATA_DIR, i) for i in os.listdir(DATA_DIR)]

_sample_size   = 10_000
inds           = generate_chunk_inds(len(files), _sample_size)
w              = get_sample_weights(inds, inds[-1][-1])



# S T E P   1   :   G A T H E R   T H E   M E A N S
mean           = np.zeros([1, siz*siz*3], dtype = np.float32)
for i, start, end in inds:
    x          = get_data_chunk(start, end, files, siz)
    mean      += w[i] * np.mean(x, 0, keepdims = True)


# S T E P   2   :   G A T H E R   T H E   S T D S
var             = np.zeros([1, siz*siz*3], dtype = np.float32)
w2              = get_sample_weights(inds, inds[-1][-1] - 1) # correct for sample var
for i, start, end in inds:
    x           = get_data_chunk(start, end, files, siz)
    var        += w2[i] * np.mean(np.square(x - mean), axis = 0, keepdims = True)
std             = np.sqrt(var)
std[std == 0.0] = 1.0

pickle.dump([mean, std], open(os.path.join(SAVE_DIR, f'PCA_5_partial_result.pkl'), 'wb'))


# S T E P   3   :   C O V A R I A N C E 
# cov             = np.zeros([siz*siz*3, siz*siz*3], dtype = np.float32)
# #cov = pickle.load(open(os.path.join(SAVE_DIR, 'PCA_6_partial_cov_40.pkl'), 'rb'))
# for i, start, end in inds:
#     #if i <= 40:
#     #    continue 
#     print(f'starting {i}')
#     x           = get_data_chunk(start, end, files, siz)
#     _x          = np.array((x - mean) / std, dtype = np.float32) # standardize the data using the mean and std
#     cov        += _x.T @ _x / (inds[-1][-1] - 1) # https://datascienceplus.com/understanding-the-covariance-matrix/
#     if i % 5 == 0:
#         pickle.dump(cov, open(os.path.join(SAVE_DIR, f'PCA_6_partial_cov_{i}.pkl'), 'wb'))

# # S T E P   4   :   P C A
# from sklearn.utils.extmath import randomized_svd
# U = randomized_svd(cov, components)[0]

# pickle.dump([siz, components, mean, std, U], open(os.path.join(SAVE_DIR, f'PCA_5_animeScrib_{siz}.pkl'), 'wb'))

mean, std = pickle.load(open(os.path.join(SAVE_DIR, f'PCA_5_partial_result.pkl'), 'rb'))



from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(n_components = components)

for i, start, end in inds:
    print(f'starting {i}')
    x           = get_data_chunk(start, end, files, siz)
    _x          = np.array((x - mean) / std, dtype = np.float32) # standardize the data using the mean and std
    ipca.partial_fit(_x)
    pickle.dump(ipca, open(os.path.join(SAVE_DIR, f'PCA_5_ipca_{i}.pkl'), 'wb'))

ipca.__dict__['singular_values_']
U = ipca.__dict__['components_'].T

pickle.dump([siz, components, mean, std, U], open(os.path.join(SAVE_DIR, f'PCA_5_celebA_recropped_{siz}.pkl'), 'wb'))








import numpy as np
import tkinter as tk
from tkinter import ttk
import zipfile, io, os, pickle
from PIL import Image, ImageTk

SAVE_DIR       = r'C:\Users\i_hat\Desktop\bastl\py\deep_larn\anime_PCA\p'


def from_latent(z, add_mean = True): 
    x = z @ U.T # comes back from latent space 
    if add_mean:
        return x * std + mean # when we transpose from the latent space, we need to unstandardize it
    return x * std

class Viewer():
    def reconstruct(self):
        code = np.array([[i.get() for i in self.vals]])
        self.x = np.clip(from_latent(code, self.add_mean),0,255).astype(np.uint8).reshape([self.siz,self.siz,3])

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
siz, components, mean, std, U = pickle.load(open(os.path.join(SAVE_DIR, 'PCA_5_anime_400_64.pkl'), 'rb'))
siz, components, mean, std, U = pickle.load(open(os.path.join(SAVE_DIR, 'PCA_5_anime_100_64.pkl'), 'rb'))
v.start(components, siz, slider_max = 100.0, std = 5.0)
siz, components, mean, std, U = pickle.load(open(os.path.join(SAVE_DIR, 'PCA_5_animal_cat_128.pkl'), 'rb'))
siz, components, mean, std, U = pickle.load(open(os.path.join(SAVE_DIR, 'PCA_5_animal_dog_128.pkl'), 'rb'))
siz, components, mean, std, U = pickle.load(open(os.path.join(SAVE_DIR, 'PCA_5_animal_wild_128.pkl'), 'rb'))
v.start(components, siz, slider_max = 200.0, std = 20.0)
siz, components, mean, std, U = pickle.load(open(os.path.join(SAVE_DIR, 'PCA_5_anime_100_128.pkl'), 'rb'))
siz, components, mean, std, U = pickle.load(open(os.path.join(SAVE_DIR, 'PCA_5_anime_100_128_recropped.pkl'), 'rb'))
siz, components, mean, std, U = pickle.load(open(os.path.join(SAVE_DIR, 'PCA_5_anime_400_128.pkl'), 'rb'))
siz, components, mean, std, U = pickle.load(open(os.path.join(SAVE_DIR, 'PCA_5_animeScrib_128.pkl'), 'rb'))
v.start(components, siz, slider_max = 200.0, std = 10.0)
siz, components, mean, std, U = pickle.load(open(os.path.join(SAVE_DIR, 'PCA_5_celebA_128.pkl'), 'rb'))
siz, components, mean, std, U = pickle.load(open(os.path.join(SAVE_DIR, 'PCA_5_celebA_recropped_128.pkl'), 'rb'))
v.start(components, siz, slider_max = 200.0, std = 20.0)






'''
PCA on 100 GB aligned anime face data. 
- i got the big dada set and this time wanted to go big - increase size from 64 to 128 pixels, and implement the full rgb covariance matrix.
- the matrix barely fits into 16gb ram and there is just enough space left to use random_svd to get the top 100 eigenvectors
- there may be a way to go without the cov matrix, but i'm not sure that's a good idea if the data can't be stored in ram at all. 
RESULTS
- the mean image even has light reflection in the eye, that's how much more detailed it is. The top features are also very blurry, 
- indicative of how much data we went through and how averaged everything is. 
- there is a lot of chin cutting in this one. im not sure if it's just more noticeable because of the better resolution, and more averaged colors.

- i think the features here are really simple to understand. each one is a specific combination of eye and face shape and color. 

- ok as we get into features 30+ is becomes clear that we are just interpolating between images as we add the feature. i think this goes to show
    how "well averaged" the first 30 features are, that we can barely tell they are images being added together (interpolated) and they work
    together seamlessly. I mean we can add the rotation feature to the smile feature and to the eye size feature and they make a new face
    that is rotated, big eyes, and has a certain hair color. I think this is because all these features have eyes in almost the same place, 
    hair in almost the same place, and the face rotation rotates it in a way that doesn't disturb the features' locations.
- i say this because in the latter features we get eyebrows that appear out of nowhere instead of smoothly rising from lower eyebrows. 
    these features are less amenable to the average, meaning they don't appear as well distributed, so they do not apply to the average face well. 

- around feature 50 they become noise which adds what seems like specific shading to the picture. maybe there was a bundle of images that had 
    similar shading, and each bundle makes up a noise feature. 

- i think this concludes my adventures with PCA on anime faces. it looks liek magic because the eigenvectors for first features are so well averaged
    that they all smooth and blend together seamlessly, but that stops as we get into further, less averaged vectors, which look like interpolating images.

- the only real problem here is the chins. why are chins so entangled with all the features? i need to check if its in the other models too. 
    in the 64x64 model it's barely noticeable. it seems there is less variance in chin size.. maybe in that data set? also the resolution really does 
    hide some of the imperfections (combinations of features which look like interpolations of images). im gonna try 100gb with 64x64 to compare. 

- ok i think jaw size is somehow getting in the way of features being disentangled. for example, features 22 and 23 are the same except for jaw size.
    similarly, features 30 and 31 are the same aside from jaw size. 
    features 36,37 are similarly entangled except with eye color. 

- if you think about why correlations happen:
    why can't PCA just find one vector with positive values where the eye pixels are, and zero everywhere else?
    well that's because in this data set, wherever they are positive there, they are also positive in certain jaw pixels (correlation)
    thus we get a feature for eye color that also changes jaw size. 
    we know this is the case because why do eye colors both change at the same time - because they are always in the same direction / same values. 
    and then i guess the PCA axes have to be orthogonal so we get the rest of the jaw feature distributed amongst otehr features. 
    what if we introduce a sparsity constraint? so eigenvectors try to have chunks of positive and zero everywhere else. but this seems like 
    feature engineering, which we want to avoid at all costs. actually it's a bad idea - think of the rotation feature which is very interrelated everywhere.
'''





'''

- - - what do i want to achieve with music?
- a groovebox to have fun with. like pocket_operator / volca_sample. -> buy OP-1 (im a nomad. its size will be useful)
- buy modules that i think sound really cool (there are too many. find job)
- create modules because that's way more interesting than learning how a module works
    - forget electronics. I know enough to use MI's IO protection circuits w MCUs
    - I won't have time/space to work with soldering any time soon. when I do, start w Hagiwo & Bastl
    - STEP 0. learn DSP w Pirkle. basic algorithms in Reaktor and VST3 * * * 


- - - - - Eurorack Modules I'd like to emulate:  
- Make Noise: tElharmonic, Erbe Verbe, Echophon <- Pirkle
- Cursus Iteritas, Loquelic Iteritas            <- Pirkle
- Bastl Timber                                  <- digital analog modeling?


- - - E L E C T R O N I X    G O A L S 
- Some Assembly Required                    <- get good at MCUs and communication protocols
- ... audio
- program the Arduino WM8731 shield         <- learn to program codec (just read the code)
- make basic utilities with Hagiwo          <- make stuff like basic drones, utilities, etc
- ... digital design
- make multi-threaded multicore processor   <- Yamin Li 
- make c compiler for this processor        <- xasm / retargetable
- make RTOS for this processor              <- udemy (second course out now), FlingOS * 




- - - scarlett solo is always 24-bit depth, and sample rate range form 44.1 to 192 kHz
    https://support.focusrite.com/hc/en-gb/articles/207546835-Is-my-Scarlett-interface-in-16-bit-or-24-bit-on-Windows-
    https://focusrite.com/en/scarlett_indepth
- - - expert sleepers es-9: 24-bit depth, sample rate 44.1 to 96 kHz

- - - cool analog circuits
- analog bit crusher (ref pittsburgh)
    https://pittsburghmodular.com/collapse
    https://pittsburghmodular.com/crush
    search: analog bit crusher // Analog Sample Rate Reducer // Analog Bitcrusher Colin Raffel
    https://www.diystompboxes.com/smfforum/index.php?topic=48809.0
    https://wraalabs.wixsite.com/pedals/single-post/2017/11/16/Sample-Rate-Reducers-Ring-Mods
    - https://en.wikipedia.org/wiki/Bitcrusher
        two effects: bit depth (not seen in analog) and sample rate (which introduces aliasing of hi freqs since no lo-pass filter)
    - https://www.strymon.net/what-is-a-bitcrusher/
        24 bit audio could theoretically deliver up to 144dB signal-to-noise, but current hardware can't deliver that 
        meaning the softest sound 0..0001 and loudest sound 11...11 span 144dB. thats why we dont see it go beyond 24-bit depth
    - based on Colin Raffel's design 
        https://shop.pedalparts.co.uk/product/fp_crusher
        https://shop.pedalparts.co.uk/product/crusher
    - https://hackaday.com/2013/08/25/the-difference-between-bitcrushers-and-sample-rate-reducers/
        so bitcrushers just reduce sample rate. reducing bit depth makes audio sound loud and worse. 
    - https://www.mojo-audio.com/blog/the-24bit-delusion/
        hardware cannot create the dynamic range of 144 dB needed for quietest/loudest possible bits in 24-bit encoding
        also we cannot hear between 0-30 dB so our ears hear only 60 dB, covered by 16-bit
        24-bits are used during editing to remove noise when audio gets downsampled to commercial resolution
    - https://modwiggler.com/forum/viewtopic.php?t=244893&sid=79dddc76ceb5261f0e232cf73d411cee&start=25
        * * * Sample rate reduction (as mentioned) can be done with a s/h, bit reduction with a quantizer.... * * * 
    AWESOME http://clsound.com/bitseuro.html



'''





'''

Tiny C Compiler

- elements of microprocessors: basic 16 bit MCU. check AVR instructions for what instr to include


- elements of C: figure out what features of C to support

- map C features to assembly chunks / templates



- try in vcv:
- sample & hold on sequencer going into pitch of VCO, should make cooler rhythms
- what the fuck is happening when you have 2 sequencers with a SH???


https://library.vcvrack.com/StudioSixPlusOne/IversonJr
https://library.vcvrack.com/SynthKit/4-StepSequencer
https://library.vcvrack.com/Bogaudio/Bogaudio-SampleHold
https://library.vcvrack.com/AS/BPMClock
https://library.vcvrack.com/Hora-treasureFree/Deep
https://library.vcvrack.com/Autodafe-DrumKit/DrumsClaps
https://library.vcvrack.com/SurgeRack/SurgeNoise
https://library.vcvrack.com/computerscare/computerscare-horse-a-doodle-doo
https://library.vcvrack.com/EricaCopies/BlackWaveTableVCO
https://library.vcvrack.com/LindenbergResearch/MS20_VCF
https://library.vcvrack.com/Atelier/AtelierPalette
https://library.vcvrack.com/squinkylabs-plug1/squinkylabs-f2
https://library.vcvrack.com/AlrightDevices/Chronoblob2
https://library.vcvrack.com/Befaco/Percall
https://library.vcvrack.com/Befaco/SpringReverb
https://library.vcvrack.com/CountModula/Mixer



'''




import os 

os.getcwd()
dir1 = os.path.join(os.getcwd(), 'my_ML/PCA/finished')
dir2 = os.path.join(os.getcwd(), 'my_ML/PCA/PCA-main')

# diff
# PCA_4_combined_rgb.py

os.listdir(dir1)
# [ '']

os.listdir(dir2)
# [ 'PCA_6_huge_data.py']

f1 = os.path.join(dir1, 'PCA_4_combined_rgb.py')
f2 = os.path.join(dir2, 'PCA_4_combined_rgb.py')
f1 == f2

_f1 = open(f1, 'r')
text1 = _f1.read()
_f1.close()


_f2 = open(f2, 'r')
text2 = _f2.read()
_f2.close()

text1 == text2