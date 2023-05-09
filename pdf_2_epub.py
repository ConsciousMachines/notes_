import pdf2image as p2i
from PIL import Image, ImageEnhance, ImageOps, ImageDraw
import numpy as np
import zipfile, os, io
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk
import PIL.Image as im



def get_computer_to_ereader_scale():
    # show a preview of how a half-page will look on my device, 
    #     by displaying a window with the same size (relative to computer's pixels)
    #     as the size of my e-reader, and showing the page in it.
    # zenbook pro 15 ux550ve :: 15.6” 1920 x 1080
    # boox note2 :: 10.3" 1872 x 1404 (227dpi)
    _diagonal_pixels = np.sqrt(1920**2 + 1080**2) # length of diagonal in pixels (pythagoras)
    _dpi = _diagonal_pixels / 15.6 # ratio of diagonal pixels to diagonal inches (dpi, ppi)
    _dpir = _dpi / 227.0 # ratio of computer dpi to e-reader dpi, to scale images to have same size
    EQ_WIDTH = int(1872 * _dpir) # this width on laptop screen is same as e-reader width irl
    EQ_HEIGHT = int(1404 * _dpir) # so we can rescale imgs to these dims to preview text size on e-reader
    return (EQ_WIDTH, EQ_HEIGHT)


EQ_WIDTH, EQ_HEIGHT = get_computer_to_ereader_scale()


def resize_to_EQ_dims(_pic, _resample_type = im.NEAREST):
    # resize the image so computer displays it exactly the same size as e-reader would display irl. 
        
    # step 1: get width and height of image, and h/w ratio
    if type(_pic) == np.array:
        _w = _pic.shape[1] # im width
        _h = _pic.shape[0] # im height
    elif type(_pic) == im.Image:
        _w = _pic.size[0]
        _h = _pic.size[1]
    else:
        raise Exception('bad image type, should be im.array or im.Image')
    _whr = _h / _w         # height/width ratio

    # step 2: calculate desired width & height, by comparing to EQ_W, EQ_H
    if _whr < (EQ_HEIGHT / EQ_WIDTH):
        # case 1: image is wider, then h/w is smaller since w is bigger. so resize to have _w = EQ_WIDTH
        _desired_w = EQ_WIDTH 
        _desired_h = int(_h * (EQ_WIDTH / _w)) # if EQ_WIDTH larger, our scale increase is (EQ_WIDTH / _w)
    else:
        # case 2: image is taller, h/w larger since h larger. resize to have h = EQ_HEIGHT
        _desired_h = EQ_HEIGHT 
        _desired_w = int(_w * (EQ_HEIGHT / _h)) # if EQ_HEIGHT larger, our scale increase is (EQ_HEIGHT/_h)

    # step 3: resize 
    if type(_pic) == np.array:
        _resized = im.fromarray(_pic).resize([_desired_w, _desired_h], resample = _resample_type)
    if type(_pic) == im.Image:
        _resized = _pic.resize([_desired_w, _desired_h], resample = _resample_type)
    else:
        raise Exception('bad image type, should be im.array or im.Image')

    return _resized


# def general_crop(p, step = 5):
#     try:
#         side_crop_left                      = 0
#         side_crop_right                     = p.shape[1]-1
#         top_crop                            = 0
#         bot_crop                            = p.shape[0]-1
#         while np.mean(p[:,side_crop_left])  == 0: 
#             side_crop_left                 += step
#         while np.mean(p[:,side_crop_right]) == 0: 
#             side_crop_right                -= step
#         while np.mean(p[top_crop,:])        == 0: 
#             top_crop                       += step
#         while np.mean(p[bot_crop,:])        == 0: 
#             bot_crop                       -= step
#         side_crop_left                      = max(0,side_crop_left)
#         side_crop_right                     = min(p.shape[1]-1, side_crop_right)
#         top_crop                            = max(0, top_crop)
#         bot_crop                            = min(p.shape[0]-1, bot_crop)
#         return p[top_crop:bot_crop, side_crop_left:side_crop_right]
#     except: # this is the case for blank pages
#         return p


def render_average_page(data): # what the average page looks like 
    try:
        BRUH           = 20 # skip BRUH front / back pages (covers are diff sizes)
        _test_page     = np.array(data[BRUH]).astype(np.uint8)
        for i in range(BRUH, len(data)-BRUH, 3):
            _test_page = np.minimum(_test_page, np.array(data[i]).astype(np.uint8)) 
        return (True, _test_page)
    except: # if the pages are all different sizes then we cant do it
        return (False, None)


class Viewer():
    def refresh(self, e):
        img                    = ImageOps.invert(self.page)
        width, height          = img.size
        # draw 
        x_left                 = self.vals[0].get()
        x_right                = self.vals[2].get()
        y_up                   = self.vals[1].get()
        y_down                 = self.vals[3].get()
        line_left              = [(x_left, 0), (x_left, height)]
        line_right             = [(width - x_right, 0), (width - x_right, height)]
        line_up                = [(0, y_up), (width, y_up)]
        line_down              = [(0, height - y_down), (width, height - y_down)]
        img1                   = ImageDraw.Draw(img)  
        img1.line(line_left,  fill ="red", width = 10)
        img1.line(line_right, fill ="red", width = 10)
        img1.line(line_up,    fill ="red", width = 10)
        img1.line(line_down,  fill ="red", width = 10)
        #resize
        if height > self.window_size:
            img = img.resize((int(self.window_size * (width / height)),self.window_size))

        self.photo = ImageTk.PhotoImage(image = img) # https://stackoverflow.com/questions/58411250/photoimage-zoom
        self.canvas_area.create_image(0,0,image = self.photo, anchor=tk.NW)
        self.canvas_area.update()

    def key_press(self, e):
        if e.keysym == 'Escape': # quit 
            return self.root.destroy()
        elif e.keysym == 'd':
            self.page_num += 1
        elif e.keysym == 'a':
            self.page_num -= 1
        self.page = data[self.page_num]
        self.refresh(0)

    def start(self, avg_page = None):
        self.page_num = len(data) // 2
        self.window_size = 1000
        if type(avg_page) == type(None):
            self.page = data[self.page_num]
        else:
            self.page = im.fromarray(avg_page)
        
        self.root                                        = tk.Tk()
        self.menu_left                                   = tk.Canvas(self.root, width=150, height = 400, bg = 'black')
        self.menu_left.grid(row = 0, column=0, sticky = 'nsew')
        sf                                               = ttk.Frame(self.menu_left)
        sf.bind("<Configure>",  lambda e: self.menu_left.configure(scrollregion = self.menu_left.bbox("all")))
        self.root.bind('<Key>', lambda x: self.key_press(x))
        self.menu_left.create_window((0, 0), window =sf, anchor="nw")

        self.vals                                        = [tk.DoubleVar() for i in range(4)]
        _labels = ['W', 'N', 'E', 'S']
        labs                                             = [ttk.Label(sf, text=_labels[i]) for i in range(4)]
        slds                                             = [None for i in range(4)]
        for i in range(4):
            slds[i]                                      = ttk.Scale(sf, from_ = 0.0, to = 2000.0, orient = 'vertical', 
                                    variable = self.vals[i], command = self.refresh, length = int(0.9 * self.window_size))
            slds[i].grid(column = i, row = 1, columnspan = 1, sticky = 'nsew')
            labs[i].grid(column = i, row = 0, columnspan = 1, sticky = 'nsew')

        self.canvas_area = tk.Canvas(self.root, width=self.window_size, height=self.window_size, bg = 'black')
        self.canvas_area.grid(row = 0, column=1, sticky = 'nsew') 
        self.root.grid_rowconfigure(1, weight = 1)
        self.root.grid_columnconfigure(1, weight = 1)
        self.refresh(0)
        self.root.mainloop()


# convert_from_path(pdf_path, dpi=200, output_folder=None, 
# first_page=None, last_page=None, fmt='ppm', jpegopt=None, 
# thread_count=1, userpw=None, use_cropbox=False, strict=False, 
# transparent=False, single_file=False, output_file=str(uuid.uuid4()), 
# poppler_path=None, grayscale=False, size=None, paths_only=False)
out_dir = r'/home/chad/Desktop'
one_pdf                                     = [os.path.join(out_dir, i) for i in os.listdir(out_dir) if i[-4:] == '.pdf'][0]
pdf_title                                   = os.path.basename(one_pdf).split('.')[0]
print(f"Starting:\n{pdf_title}")
data = p2i.convert_from_path(one_pdf, fmt='jpg', thread_count=os.cpu_count(), dpi = 300)
success, avg_page = render_average_page(data)
print(success)


v  = Viewer()
v.start(avg_page)
x_left                 = int(v.vals[0].get())
x_right                = int(v.vals[2].get())
y_up                   = int(v.vals[1].get())
y_down                 = int(v.vals[3].get())
print(x_left, x_right, y_up, y_down)




if False:
    pass

    ########################################### TEST ZONE ############################
    ########################################### TEST ZONE ############################
    ########################################### TEST ZONE ############################
    ########################################### TEST ZONE ############################

    # out_dir = r'/home/chad/Desktop'
    # one_pdf                                     = [os.path.join(out_dir, i) for i in os.listdir(out_dir) if i[-4:] == '.pdf'][0]
    # pdf_title                                   = os.path.basename(one_pdf).split('.')[0]
    # print(f"Starting:\n{pdf_title}")
    # data = p2i.convert_from_path(one_pdf, fmt='jpg', thread_count=os.cpu_count(), dpi = 300)
    # success, avg_page = render_average_page(data)


    # v  = Viewer()
    # v.start(avg_page)
    # x_left                 = int(v.vals[0].get())
    # x_right                = int(v.vals[2].get())
    # y_up                   = int(v.vals[1].get())
    # y_down                 = int(v.vals[3].get())
    # print(x_left, x_right, y_up, y_down)

    # ######################################################

    # i = len(data) // 2 + 10
    # width, height = data[i].size
    # _cropped = data[i].convert('L').crop((x_left + 1, y_up + 1, width - x_right, height - y_down))
    # _enhanced = ImageEnhance.Contrast(_cropped).enhance(4.)
    # _rot = _enhanced.transpose(im.ROTATE_90)

    # leniance = 20
    # width, height = _rot.size
    # _one_slice = _rot.crop((0,0,width // 2 + leniance + 1,height))
    # _two_slice = _rot.crop((width // 2 - leniance,0,width,height))

    # _one_slice = im.fromarray(255-np.array(_one_slice))
    # _pic = _one_slice.transpose(im.ROTATE_270)
    # resize_to_EQ_dims(_pic).show()










    # # function to copy text to clipboard (linux)
    # import subprocess
    # def copy_to_clipboard(text):
    #     p = subprocess.Popen(['xsel', '-bi'], stdin=subprocess.PIPE)
    #     p.communicate(input=text.encode())


    # soy = r'''


    # a. If the genotype makes its first appearance on the 53rd subject analyzed, then the first 52 subjects do not have the genotype, and the 53rd subject has the genotype. Assuming the subjects are independent and have the same prevalence probability $\theta$, the likelihood function can be modeled using a geometric distribution. The probability mass function (PMF) of a geometric distribution is:

    # $P(X = k) = (1 - \theta)^{(k - 1)} \cdot \theta$

    # In this case, $k = 53$. So the likelihood function $L(\theta)$ is:

    # $L(\theta) = (1 - \theta)^{(53 - 1)} \cdot \theta$

    # b. If the scientists had planned to stop when they found five subjects with the genotype of interest, and they analyzed 552 subjects, we can model this using a negative binomial distribution. The PMF of a negative binomial distribution is:

    # $P(X = k) = C(k - 1, r - 1) \cdot \theta^r \cdot (1 - \theta)^{(k - r)}$

    # In this case, $r = 5$ (the number of successes or genotypes of interest), and $k = 552$ (the number of trials). So the likelihood function $L(\theta)$ is:

    # $L(\theta) = C(552 - 1, 5 - 1) \cdot \theta^5 \cdot (1 - \theta)^{(552 - 5)}$

    # c. We can plot both likelihood functions in R:

    # The plot will show the likelihood functions for both scenarios a and b. You will notice that the likelihood function in scenario a, where the genotype appears on the 53rd subject, is more spread out with a lower peak than the likelihood function in scenario b, where the scientists stop after finding five subjects with the genotype. This indicates that the data in scenario b provides more information about the prevalence probability $\theta$, resulting in a more concentrated likelihood function around the most likely value of $\theta$.

    # '''

    # soy = soy.replace('θ', '$\\theta$').replace('\n', '\n\n')

    # print(soy)
    # copy_to_clipboard(soy)





    ########################################### END TEST ZONE ############################
    ########################################### END TEST ZONE ############################
    ########################################### END TEST ZONE ############################
    ########################################### END TEST ZONE ############################


# # convert a directory of scans
# pdf_title = 'zhu_embedded'
# scan_dir = r'/home/chad/Desktop/media/zhu'

# out_dir = r'/home/chad/Desktop'
# x_left                 = 0
# x_right                = 0
# y_up                   = 0
# y_down                 = 0
# scan_files = [os.path.join(scan_dir, i) for i in os.listdir(scan_dir)]
# scan_files = sorted(scan_files)
# data = []
# for i in scan_files:
#     data.append(im.open(i))

# c o n t r a s t   &   s a v e
with zipfile.ZipFile(os.path.join(out_dir, f'{pdf_title}.cbz'), 'w') as zf:

    # write the original cover
    file_object = io.BytesIO() # https://stackoverflow.com/questions/63439403/how-to-create-a-zip-file-in-memory-with-a-list-of-pil-image-objects
    data[0].save(file_object, 'jpeg')
    zf.writestr('0000.jpg', file_object.getvalue())

    for i in range(len(data)):

        # crop
        width, height = data[i].size
        _cropped = data[i].convert('L').crop((x_left + 1, y_up + 1, width - x_right, height - y_down))

        # enhance contrast
        _enhanced = ImageEnhance.Contrast(_cropped).enhance(4.)

        # rotate 90 deg 
        _rot = _enhanced.transpose(im.ROTATE_90)

        # split 
        leniance = 20
        width, height = _rot.size
        _one_slice = _rot.crop((0,0,width // 2 + leniance + 1,height))
        _two_slice = _rot.crop((width // 2 - leniance,0,width,height))

        # write
        _ = file_object.seek(0)
        _ = file_object.truncate(0)
        _one_slice.save(file_object, 'jpeg')
        zf.writestr(f"{str(i*2 + 1).rjust(4, '0')}.jpg", file_object.getvalue())
        _ = file_object.seek(0)
        _ = file_object.truncate(0)
        _two_slice.save(file_object, 'jpeg')
        zf.writestr(f"{str(i*2 + 2).rjust(4, '0')}.jpg", file_object.getvalue())
        
        # memory management, otherwise process gets killed
        data[i] = None 




import os 
import subprocess


libs = [i.strip() for i in '''dctrl-tools
dkms
glx-alternative-mesa
glx-alternative-nvidia
glx-diversions
libcuda1:amd64
libegl-nvidia0:amd64
libgl1-nvidia-glvnd-glx:amd64
libgles-nvidia1:amd64
libgles-nvidia2:amd64
libgles1:amd64
libglx-nvidia0:amd64
libnvcuvid1:amd64
libnvidia-cbl:amd64
libnvidia-cfg1:amd64
libnvidia-egl-wayland1:amd64
libnvidia-eglcore:amd64
libnvidia-encode1:amd64
libnvidia-glcore:amd64
libnvidia-glvkspirv:amd64
libnvidia-ml1:amd64
libnvidia-ptxjitcompiler1:amd64
libnvidia-rtcore:amd64
libopengl0:amd64
libxnvctrl0:amd64
linux-headers-5.10.0-22-amd64
linux-headers-5.10.0-22-common
linux-headers-amd64
nvidia-alternative
nvidia-driver
nvidia-driver-bin
nvidia-driver-libs:amd64
nvidia-egl-common
nvidia-egl-icd:amd64
nvidia-installer-cleanup
nvidia-kernel-common
nvidia-kernel-dkms
nvidia-kernel-support
nvidia-legacy-check
nvidia-modprobe
nvidia-persistenced
nvidia-settings
nvidia-smi
nvidia-support
nvidia-vdpau-driver:amd64
nvidia-vulkan-common
nvidia-vulkan-icd:amd64
update-glx
xserver-xorg-video-nvidia'''.split('\n')]


apt_cache_dir = '/var/cache/apt/archives'
backup_dir = '/home/chad/Desktop/nv_driver_cache'


for i in libs:
    cmd = f'sudo mv {os.path.join(apt_cache_dir, i)} {os.path.join(backup_dir, i)}'.strip().split(' ')
    if subprocess.call(cmd) != 0:
        print(f'FAILED: {i}')


'''
sudo apt remove --purge $(cat packages_to_remove.txt)
sudo apt remove nvidia-kernel-dkms
'''


import torch

torch.cuda.is_available()

torch.cuda.device_count()

torch.cuda.current_device()

torch.cuda.device(0)

torch.cuda.get_device_name(0)

device = torch.device('cuda')

a = torch.tensor([1,2,3,4], dtype=torch.float32).to(device)
b = torch.tensor([1,2,3,4], dtype=torch.float32).to(device)
c = a @ b 
c




# Harry Potter and the cursed Fan
# 1. ever since I started using linux, everything was fantastic except for one issue:
#       when the computer is at idle, the fan ramps up to max RPM and keeps going (nothing is running, room temp)
#       the only way to avoid this event is to keep Youtube playing from Firefox (other browsers, or playing videos locally wont work)
# 2. a year later, I noticed this issue doesn't happen on fresh installs anymore except until I install nvidia drivers.
#       I checked what packages exist before and after the nvidia driver installation. 
#       removing all of them fixes the fan issue. but i wanted to see if i can remove just one, to act as a 
#       way to 'turn off' the driver without completely uninstalling everything. 
# 3. seems 'nvidia-kernel-dkms' is one such package. uninstalling is breaks the driver, fixing the fan issue.
#       when i need to use pytorch i can just re-install this small package. 
#       when i remove this package alone, it takes 'nvidia-driver' along with it, 50 MB. 
#       now i can just download the .deb for these 2 packages and install that! the download is 27 MB
#
#       sudo apt download nvidia-driver nvidia-kernel-dkms
#       sudo dpkg -i nvidia-driver_470.182.03-1_amd64.deb  
#       sudo dpkg -i nvidia-kernel-dkms_470.182.03-1_amd64.deb
#
# trying other packages because installing dkms takes time.
# 4. I basically uninstalled one package at a time and noticed when the fan starts vs stops. it seems the offending package is:
#       libglx-nvidia0:amd64
#       and without it, I can use CUDA without uninstalling the driver!!!
# also, i wonder if the problem still exists on Arch / PopOS / Mint ???





import os

fan_dir = r'/home/chad/Desktop/_backups/backups/soy_fan/part2'
files = os.listdir(fan_dir)
files = ' '.join(files)


packages = [i.strip().replace(':amd64','') for i in '''
dctrl-tools
dkms
glx-alternative-mesa
glx-alternative-nvidia
glx-diversions
libcuda1:amd64
libegl-nvidia0:amd64
libgl1-nvidia-glvnd-glx:amd64
libgles-nvidia1:amd64
libgles-nvidia2:amd64
libgles1:amd64
libglx-nvidia0:amd64
libnvcuvid1:amd64
libnvidia-cbl:amd64
libnvidia-cfg1:amd64
libnvidia-egl-wayland1:amd64
libnvidia-eglcore:amd64
libnvidia-encode1:amd64
libnvidia-glcore:amd64
libnvidia-glvkspirv:amd64
libnvidia-ml1:amd64
libnvidia-ptxjitcompiler1:amd64
libnvidia-rtcore:amd64
libopengl0:amd64
libxnvctrl0:amd64
linux-headers-5.10.0-22-amd64
linux-headers-5.10.0-22-common
linux-headers-amd64
nvidia-alternative
nvidia-driver
nvidia-driver-bin
nvidia-driver-libs:amd64
nvidia-egl-common
nvidia-egl-icd:amd64
nvidia-installer-cleanup
nvidia-kernel-common
nvidia-kernel-dkms
nvidia-kernel-support
nvidia-legacy-check
nvidia-modprobe
nvidia-persistenced
nvidia-settings
nvidia-smi
nvidia-support
nvidia-vdpau-driver:amd64
nvidia-vulkan-common
nvidia-vulkan-icd:amd64
update-glx
xserver-xorg-video-nvidia
'''.split('\n') if len(i) > 2]

len(packages)
packages = sorted(packages)

for p in packages:
    print(p)

_all = True
for p in packages:
    _all = _all and (p in files)
_all
    





import os
import subprocess

fan_dir = r'/home/chad/Desktop/_backups/backups/soy_fan/all'
os.chdir(fan_dir)

# get the actual .deb files
result = subprocess.run(['ls'], stdout=subprocess.PIPE)
output = result.stdout.decode('utf-8')
files = [i.strip() for i in output.split('\n') if len(i) > 2]
files 

for f in files:
    cmd = f'dpkg -I {f} | grep Depends'#.split(' ')
    print(cmd)
    # result = subprocess.run(cmd, stdout=subprocess.PIPE)
    # output = result.stdout.decode('utf-8')
    # print(output)






'''
dpkg -I linux-headers-5.10.0-22-common_5.10.178-3_all.deb | grep Depends
dpkg -I linux-headers-5.10.0-22-amd64_5.10.178-3_amd64.deb | grep Depends
dpkg -I linux-headers-amd64_5.10.178-3_amd64.deb | grep Depends
dpkg -I nvidia-installer-cleanup_20151021+13_amd64.deb | grep Depends
dpkg -I nvidia-legacy-check_470.182.03-1_amd64.deb | grep Depends
dpkg -I dctrl-tools_2.24-3+b1_amd64.deb | grep Depends
dpkg -I dkms_2.8.4-3_all.deb | grep Depends
dpkg -I nvidia-modprobe_470.182.03-1_amd64.deb | grep Depends
dpkg -I libnvidia-rtcore_470.182.03-1_amd64.deb | grep Depends
dpkg -I nvidia-support_20151021+13_amd64.deb | grep Depends
dpkg -I libnvidia-ptxjitcompiler1_470.182.03-1_amd64.deb | grep Depends
dpkg -I nvidia-kernel-common_20151021+13_amd64.deb | grep Depends
dpkg -I libnvidia-cbl_470.182.03-1_amd64.deb | grep Depends
dpkg -I libxnvctrl0_470.141.03-1~deb11u1_amd64.deb | grep Depends
dpkg -I update-glx_1.2.1~deb11u1_amd64.deb | grep Depends
dpkg -I glx-alternative-mesa_1.2.1~deb11u1_amd64.deb | grep Depends
dpkg -I glx-diversions_1.2.1~deb11u1_amd64.deb | grep Depends
dpkg -I glx-alternative-nvidia_1.2.1~deb11u1_amd64.deb | grep Depends
dpkg -I nvidia-alternative_470.182.03-1_amd64.deb | grep Depends
dpkg -I libnvidia-glcore_470.182.03-1_amd64.deb | grep Depends
dpkg -I libnvidia-glvkspirv_470.182.03-1_amd64.deb | grep Depends
dpkg -I libopengl0_1.3.2-1_amd64.deb | grep Depends
dpkg -I libnvidia-eglcore_470.182.03-1_amd64.deb | grep Depends
dpkg -I nvidia-egl-common_470.182.03-1_amd64.deb | grep Depends
dpkg -I libnvidia-egl-wayland1_1%3a1.1.5-1_amd64.deb | grep Depends
dpkg -I libegl-nvidia0_470.182.03-1_amd64.deb | grep Depends
dpkg -I nvidia-egl-icd_470.182.03-1_amd64.deb | grep Depends
dpkg -I libglx-nvidia0_470.182.03-1_amd64.deb | grep Depends
dpkg -I libgl1-nvidia-glvnd-glx_470.182.03-1_amd64.deb | grep Depends
dpkg -I nvidia-vdpau-driver_470.182.03-1_amd64.deb | grep Depends
dpkg -I nvidia-vulkan-common_470.182.03-1_amd64.deb | grep Depends
dpkg -I nvidia-vulkan-icd_470.182.03-1_amd64.deb | grep Depends
dpkg -I libgles1_1.3.2-1_amd64.deb | grep Depends
dpkg -I libgles-nvidia1_470.182.03-1_amd64.deb | grep Depends
dpkg -I libgles-nvidia2_470.182.03-1_amd64.deb | grep Depends
dpkg -I libnvidia-ml1_470.182.03-1_amd64.deb | grep Depends
dpkg -I nvidia-smi_470.182.03-1_amd64.deb | grep Depends
dpkg -I libnvidia-cfg1_470.182.03-1_amd64.deb | grep Depends
dpkg -I libcuda1_470.182.03-1_amd64.deb | grep Depends
dpkg -I nvidia-driver-libs_470.182.03-1_amd64.deb | grep Depends
dpkg -I nvidia-driver-bin_470.182.03-1_amd64.deb | grep Depends
dpkg -I libnvcuvid1_470.182.03-1_amd64.deb | grep Depends
dpkg -I libnvidia-encode1_470.182.03-1_amd64.deb | grep Depends
dpkg -I nvidia-kernel-support_470.182.03-1_amd64.deb | grep Depends
dpkg -I nvidia-kernel-dkms_470.182.03-1_amd64.deb | grep Depends
dpkg -I nvidia-settings_470.141.03-1~deb11u1_amd64.deb | grep Depends
dpkg -I nvidia-persistenced_470.103.01-2~deb11u1_amd64.deb | grep Depends
dpkg -I xserver-xorg-video-nvidia_470.182.03-1_amd64.deb | grep Depends
dpkg -I nvidia-driver_470.182.03-1_amd64.deb | grep Depends

'''










'''
linux-headers-5.10.0-22-common_5.10.178-3_all.deb
linux-headers-5.10.0-22-amd64_5.10.178-3_amd64.deb
linux-headers-amd64_5.10.178-3_amd64.deb
nvidia-installer-cleanup_20151021+13_amd64.deb
nvidia-legacy-check_470.182.03-1_amd64.deb
dctrl-tools_2.24-3+b1_amd64.deb
dkms_2.8.4-3_all.deb
nvidia-modprobe_470.182.03-1_amd64.deb
libnvidia-rtcore_470.182.03-1_amd64.deb
nvidia-support_20151021+13_amd64.deb
libnvidia-ptxjitcompiler1_470.182.03-1_amd64.deb
nvidia-kernel-common_20151021+13_amd64.deb
libnvidia-cbl_470.182.03-1_amd64.deb
libxnvctrl0_470.141.03-1~deb11u1_amd64.deb
update-glx_1.2.1~deb11u1_amd64.deb
glx-alternative-mesa_1.2.1~deb11u1_amd64.deb
glx-diversions_1.2.1~deb11u1_amd64.deb
glx-alternative-nvidia_1.2.1~deb11u1_amd64.deb
nvidia-alternative_470.182.03-1_amd64.deb
libnvidia-glcore_470.182.03-1_amd64.deb
libnvidia-glvkspirv_470.182.03-1_amd64.deb
libopengl0_1.3.2-1_amd64.deb
libnvidia-eglcore_470.182.03-1_amd64.deb
nvidia-egl-common_470.182.03-1_amd64.deb
libnvidia-egl-wayland1_1%3a1.1.5-1_amd64.deb
libegl-nvidia0_470.182.03-1_amd64.deb
nvidia-egl-icd_470.182.03-1_amd64.deb
libglx-nvidia0_470.182.03-1_amd64.deb
libgl1-nvidia-glvnd-glx_470.182.03-1_amd64.deb
nvidia-vdpau-driver_470.182.03-1_amd64.deb
nvidia-vulkan-common_470.182.03-1_amd64.deb
nvidia-vulkan-icd_470.182.03-1_amd64.deb
libgles1_1.3.2-1_amd64.deb
libgles-nvidia1_470.182.03-1_amd64.deb
libgles-nvidia2_470.182.03-1_amd64.deb
libnvidia-ml1_470.182.03-1_amd64.deb
nvidia-smi_470.182.03-1_amd64.deb
libnvidia-cfg1_470.182.03-1_amd64.deb
libcuda1_470.182.03-1_amd64.deb
nvidia-driver-libs_470.182.03-1_amd64.deb
nvidia-driver-bin_470.182.03-1_amd64.deb
libnvcuvid1_470.182.03-1_amd64.deb
libnvidia-encode1_470.182.03-1_amd64.deb
nvidia-kernel-support_470.182.03-1_amd64.deb
nvidia-kernel-dkms_470.182.03-1_amd64.deb
nvidia-settings_470.141.03-1~deb11u1_amd64.deb
nvidia-persistenced_470.103.01-2~deb11u1_amd64.deb
xserver-xorg-video-nvidia_470.182.03-1_amd64.deb
nvidia-driver_470.182.03-1_amd64.deb
'''


'''
sudo dpkg -i linux-headers-5.10.0-22-common_5.10.178-3_all.deb linux-headers-5.10.0-22-amd64_5.10.178-3_amd64.deb linux-headers-amd64_5.10.178-3_amd64.deb nvidia-installer-cleanup_20151021+13_amd64.deb nvidia-legacy-check_470.182.03-1_amd64.deb dctrl-tools_2.24-3+b1_amd64.deb dkms_2.8.4-3_all.deb nvidia-modprobe_470.182.03-1_amd64.deb libnvidia-rtcore_470.182.03-1_amd64.deb nvidia-support_20151021+13_amd64.deb libnvidia-ptxjitcompiler1_470.182.03-1_amd64.deb nvidia-kernel-common_20151021+13_amd64.deb libnvidia-cbl_470.182.03-1_amd64.deb libxnvctrl0_470.141.03-1~deb11u1_amd64.deb update-glx_1.2.1~deb11u1_amd64.deb glx-alternative-mesa_1.2.1~deb11u1_amd64.deb glx-diversions_1.2.1~deb11u1_amd64.deb glx-alternative-nvidia_1.2.1~deb11u1_amd64.deb nvidia-alternative_470.182.03-1_amd64.deb libnvidia-glcore_470.182.03-1_amd64.deb libnvidia-glvkspirv_470.182.03-1_amd64.deb libopengl0_1.3.2-1_amd64.deb libnvidia-eglcore_470.182.03-1_amd64.deb nvidia-egl-common_470.182.03-1_amd64.deb libnvidia-egl-wayland1_1%3a1.1.5-1_amd64.deb libegl-nvidia0_470.182.03-1_amd64.deb nvidia-egl-icd_470.182.03-1_amd64.deb libglx-nvidia0_470.182.03-1_amd64.deb libgl1-nvidia-glvnd-glx_470.182.03-1_amd64.deb nvidia-vdpau-driver_470.182.03-1_amd64.deb nvidia-vulkan-common_470.182.03-1_amd64.deb nvidia-vulkan-icd_470.182.03-1_amd64.deb libgles1_1.3.2-1_amd64.deb libgles-nvidia1_470.182.03-1_amd64.deb libgles-nvidia2_470.182.03-1_amd64.deb libnvidia-ml1_470.182.03-1_amd64.deb nvidia-smi_470.182.03-1_amd64.deb libnvidia-cfg1_470.182.03-1_amd64.deb libcuda1_470.182.03-1_amd64.deb nvidia-driver-libs_470.182.03-1_amd64.deb nvidia-driver-bin_470.182.03-1_amd64.deb libnvcuvid1_470.182.03-1_amd64.deb libnvidia-encode1_470.182.03-1_amd64.deb nvidia-kernel-support_470.182.03-1_amd64.deb nvidia-kernel-dkms_470.182.03-1_amd64.deb nvidia-settings_470.141.03-1~deb11u1_amd64.deb nvidia-persistenced_470.103.01-2~deb11u1_amd64.deb xserver-xorg-video-nvidia_470.182.03-1_amd64.deb nvidia-driver_470.182.03-1_amd64.deb
'''













'''
sudo dpkg -i linux-headers-5.10.0-22-common_5.10.178-3_all.deb
sudo dpkg -i linux-headers-5.10.0-22-amd64_5.10.178-3_amd64.deb
sudo dpkg -i linux-headers-amd64_5.10.178-3_amd64.deb
sudo dpkg -i nvidia-installer-cleanup_20151021+13_amd64.deb
sudo dpkg -i nvidia-legacy-check_470.182.03-1_amd64.deb
sudo dpkg -i dctrl-tools_2.24-3+b1_amd64.deb
sudo dpkg -i dkms_2.8.4-3_all.deb
sudo dpkg -i nvidia-modprobe_470.182.03-1_amd64.deb
sudo dpkg -i libnvidia-rtcore_470.182.03-1_amd64.deb
sudo dpkg -i nvidia-support_20151021+13_amd64.deb
sudo dpkg -i libnvidia-ptxjitcompiler1_470.182.03-1_amd64.deb
sudo dpkg -i nvidia-kernel-common_20151021+13_amd64.deb
sudo dpkg -i libnvidia-cbl_470.182.03-1_amd64.deb
sudo dpkg -i libxnvctrl0_470.141.03-1~deb11u1_amd64.deb
sudo dpkg -i update-glx_1.2.1~deb11u1_amd64.deb
sudo dpkg -i glx-alternative-mesa_1.2.1~deb11u1_amd64.deb
sudo dpkg -i glx-diversions_1.2.1~deb11u1_amd64.deb
sudo dpkg -i glx-alternative-nvidia_1.2.1~deb11u1_amd64.deb
sudo dpkg -i nvidia-alternative_470.182.03-1_amd64.deb
sudo dpkg -i libnvidia-glcore_470.182.03-1_amd64.deb
sudo dpkg -i libnvidia-glvkspirv_470.182.03-1_amd64.deb
sudo dpkg -i libopengl0_1.3.2-1_amd64.deb
sudo dpkg -i libnvidia-eglcore_470.182.03-1_amd64.deb
sudo dpkg -i nvidia-egl-common_470.182.03-1_amd64.deb
sudo dpkg -i libnvidia-egl-wayland1_1%3a1.1.5-1_amd64.deb
sudo dpkg -i libegl-nvidia0_470.182.03-1_amd64.deb
sudo dpkg -i nvidia-egl-icd_470.182.03-1_amd64.deb
sudo dpkg -i libglx-nvidia0_470.182.03-1_amd64.deb
sudo dpkg -i libgl1-nvidia-glvnd-glx_470.182.03-1_amd64.deb
sudo dpkg -i nvidia-vdpau-driver_470.182.03-1_amd64.deb
sudo dpkg -i nvidia-vulkan-common_470.182.03-1_amd64.deb
sudo dpkg -i nvidia-vulkan-icd_470.182.03-1_amd64.deb
sudo dpkg -i libgles1_1.3.2-1_amd64.deb
sudo dpkg -i libgles-nvidia1_470.182.03-1_amd64.deb
sudo dpkg -i libgles-nvidia2_470.182.03-1_amd64.deb
sudo dpkg -i libnvidia-ml1_470.182.03-1_amd64.deb
sudo dpkg -i nvidia-smi_470.182.03-1_amd64.deb
sudo dpkg -i libnvidia-cfg1_470.182.03-1_amd64.deb
sudo dpkg -i libcuda1_470.182.03-1_amd64.deb
sudo dpkg -i nvidia-driver-libs_470.182.03-1_amd64.deb
sudo dpkg -i nvidia-driver-bin_470.182.03-1_amd64.deb
sudo dpkg -i libnvcuvid1_470.182.03-1_amd64.deb
sudo dpkg -i libnvidia-encode1_470.182.03-1_amd64.deb
sudo dpkg -i nvidia-kernel-support_470.182.03-1_amd64.deb
sudo dpkg -i nvidia-kernel-dkms_470.182.03-1_amd64.deb
sudo dpkg -i nvidia-settings_470.141.03-1~deb11u1_amd64.deb
sudo dpkg -i nvidia-persistenced_470.103.01-2~deb11u1_amd64.deb
sudo dpkg -i xserver-xorg-video-nvidia_470.182.03-1_amd64.deb
sudo dpkg -i nvidia-driver_470.182.03-1_amd64.deb
'''




'''
part 1
nvidia-kernel-dkms nvidia-kernel-common nvidia-kernel-support nvidia-modprobe dctrl-tools dkms nvidia-legacy-check update-glx glx-alternative-nvidia glx-diversions glx-alternative-mesa nvidia-installer-cleanup nvidia-alternative

part 2
linux-headers-amd64 linux-headers-5.10.0-22-amd64 linux-headers-5.10.0-22-common nvidia-driver nvidia-driver-bin nvidia-smi nvidia-settings nvidia-persistenced nvidia-driver-libs:amd64 libnvidia-cbl:amd64 libnvidia-cfg1:amd64 libnvidia-ml1:amd64 libxnvctrl0:amd64 libnvidia-rtcore:amd64 xserver-xorg-video-nvidia libnvidia-egl-wayland1:amd64 libegl-nvidia0:amd64 libnvidia-eglcore:amd64 libnvidia-glvkspirv:amd64 libopengl0:amd64 nvidia-vdpau-driver:amd64 libnvidia-glcore:amd64 libgles-nvidia1:amd64 libgles-nvidia2:amd64 libgles1:amd64 nvidia-egl-common nvidia-egl-icd:amd64 libgl1-nvidia-glvnd-glx:amd64 libglx-nvidia0:amd64 libnvidia-cbl libnvidia-rtcore nvidia-vulkan-icd nvidia-vulkan-common libnvidia-encode1:amd64 libnvcuvid1:amd64 libcuda1:amd64 nvidia-support libnvidia-ptxjitcompiler1:amd64 

all
nvidia-kernel-dkms nvidia-kernel-common nvidia-kernel-support nvidia-modprobe dctrl-tools dkms nvidia-legacy-check update-glx glx-alternative-nvidia glx-diversions glx-alternative-mesa nvidia-installer-cleanup nvidia-alternative linux-headers-amd64 linux-headers-5.10.0-22-amd64 linux-headers-5.10.0-22-common nvidia-driver nvidia-driver-bin nvidia-smi nvidia-settings nvidia-persistenced nvidia-driver-libs:amd64 libnvidia-cbl:amd64 libnvidia-cfg1:amd64 libnvidia-ml1:amd64 libxnvctrl0:amd64 libnvidia-rtcore:amd64 xserver-xorg-video-nvidia libnvidia-egl-wayland1:amd64 libegl-nvidia0:amd64 libnvidia-eglcore:amd64 libnvidia-glvkspirv:amd64 libopengl0:amd64 nvidia-vdpau-driver:amd64 libnvidia-glcore:amd64 libgles-nvidia1:amd64 libgles-nvidia2:amd64 libgles1:amd64 nvidia-egl-common nvidia-egl-icd:amd64 libgl1-nvidia-glvnd-glx:amd64 libglx-nvidia0:amd64 libnvidia-cbl libnvidia-rtcore nvidia-vulkan-icd nvidia-vulkan-common libnvidia-encode1:amd64 libnvcuvid1:amd64 libcuda1:amd64 nvidia-support libnvidia-ptxjitcompiler1:amd64 

'''

'''
well once again it seems nothing worked. i have 2 options: remove nvidia-kernel-dkms which is 50 MB and takes 52 seconds to reinstall, or remove 
all the packages of part 2 which are 300 MB and take 48 seconds to reinstall. I guess I'll go with dkms
'''