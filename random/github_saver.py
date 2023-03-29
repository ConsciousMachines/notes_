import os, glob, shutil


save_dirs = r'C:\Users\i_hat\Desktop\bastl'

exts_to_save = ['.py', '.vcv']
exts_to_remember = ['pdf', 'epub']

'''
linux cpp dev env
block vscode from sending info
linux reaktor

'''


files = glob.glob(os.path.join(save_dirs, '*'))
files 


result_dir = r'C:\Users\i_hat\Desktop'
os.mkdir(os.path.join(result_dir, 'result'))

to_remember = []



def walk(middle_dir = ''):
    for i in files:
        if os.path.isdir(i):
            walk(i)
        elif os.path.isfile(i):
            ext = i.split('.')[-1]
            file_name = os.path.basename(i)
            if ext in exts_to_remember:
                to_remember.append(file_name)
            if ext in exts_to_save:
                new_path = os.path.join(result_dir, file_name)
                shutil.copy(i, new_path)
        else:
            raise Exception('wtf man')









# note: the new version doesn't bother with extensions. some libs are copied, 
# since i put them in a visible position for good reason anyways!
import os, glob, shutil 

def make_dir(x): 
    if not os.path.exists(x):
        os.mkdir(x) 

work_dir = r'C:\Users\i_hat\Desktop'
solution_dir = r'C:\Users\i_hat\Desktop\bastl'
temp_dir = r'C:\Users\i_hat\Desktop\result'
make_dir(temp_dir) # make dir where the files shall go 

def my_walk(folder):
    all_paths = [os.path.join(folder, i) for i in os.listdir(folder)]
    for path in all_paths:
        name = os.path.basename(path)
        root = os.path.dirname(path)
        if os.path.isdir(path) == True:
            if name[0] == '.': continue # hidden folders 
            if name in ['Debug','Release','bin','obj','x64']: continue # intermediate folders
            my_walk(path)
        elif os.path.isfile(path) == True:
            # dest = os.path.join(temp_dir, root[len(solution_dir):].strip('\\'),name)
            # try:
            #     _ = shutil.copy(path, dest)
            # except IOError:
            #     os.makedirs(os.path.dirname(dest))
            #     _ = shutil.copy(path, dest)

            ext = path.split('.')[-1]
            file_name = os.path.basename(path)
            if ext in exts_to_remember:
                to_remember.append(file_name)
            if ext in exts_to_save:
                new_path = os.path.join(result_dir, file_name)
                shutil.copy(path, new_path)
        else:
            raise Exception('wtf man')

my_walk(solution_dir)
result_dir
file_name


# conversely, i can save a load of space by deleting the junk from repos:
def delete_interm(folder):
    all_paths = [os.path.join(folder, i) for i in os.listdir(folder)]
    for path in all_paths:
        if os.path.isdir(path) == True:
            name = os.path.basename(path)
            print(path)
            if name[0] == '.': 
                shutil.rmtree(path)
            elif name in ['Debug','Release','bin','obj','x64']: 
                shutil.rmtree(path)
            else:
                delete_interm(path)

delete_interm(solution_dir)
# 8.32 GB -> 60 MB