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
data = p2i.convert_from_path(one_pdf, fmt='jpg', thread_count=os.cpu_count(),
    dpi = 300, last_page=150)
success, avg_page = render_average_page(data)
print(success)


v  = Viewer()
v.start(avg_page)
x_left                 = int(v.vals[0].get())
x_right                = int(v.vals[2].get())
y_up                   = int(v.vals[1].get())
y_down                 = int(v.vals[3].get())
print(x_left, x_right, y_up, y_down)






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



########################################### END TEST ZONE ############################
########################################### END TEST ZONE ############################
########################################### END TEST ZONE ############################
########################################### END TEST ZONE ############################




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

