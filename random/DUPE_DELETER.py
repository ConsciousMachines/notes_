import os 
import shutil

class uuid:
    i = 0
    def get(self):
        self.i += 1
        return str(self.i)

class counter: 
    i = 0

# part 1: get size & name of all files. 
# ============================================================================
# ============================================================================

dirr = r'/home/chad/Desktop/OLD_LIB'

num_folders = counter()

# my personalized version of os.walk
def my_walk(dirr, fptr, c):

    # iterate over the folders and files in given directory
    for local_file in [os.path.join(dirr, i) for i in os.listdir(dirr)]:

        # if it's a folder
        if os.path.isdir(local_file):

            # increment folder counter 
            c.i += 1

            # recursive call 
            my_walk(local_file, fptr, c)

        # if its a file 
        if os.path.isfile(local_file):

            # get its size
            size = os.path.getsize(local_file)

            # write the data to the file 
            fptr.write(f'{size} SOY {local_file}\n')

f = open('test.txt', 'w')
my_walk(dirr, f, num_folders)
f.close()


# part 2: build a dictionary
# ============================================================================
# ============================================================================

d = {}

f = open('test.txt', 'r')

for line in f:
    parts = line.strip().split(' SOY ')
    assert len(parts) == 2, 'bruh moment'
    size, name = parts 
    if size in d.keys():
        d[size].append(name)
    else:
        d[size] = [name]

num_files = 0
for key in d.keys():
    for filee in d[key]:
        num_files += 1
# check that the number of files & folders matches what the system says
num_folders.i + num_files

# part 3: delete entries with just 1 occurence
# ============================================================================
# ============================================================================


# make a list of what to delete first, so we dont change dic size during iteration
to_delete = [] 
for k in d.keys():
    if len(d[k]) == 1:
        to_delete.append(k)


for k in to_delete:
    _ = d.pop(k)


# part 4: iterate over files of same size and check if files match. if so, delete
# ============================================================================
# ============================================================================

# list of tuples of copies x = y = z -> [(x,y),(y,z)]
copies_found = []

u = uuid()

# for each key/file-size
for k in d.keys():

    # all files of the same size. sort by length, which likely corresponds to depth
    files = sorted(d[k], key=len)

    # for each file, compare it to all the files after it
    for i in range(len(files) - 1): # all but the last 
        file1 = files[i]

        for j in range(i+1, len(files)): # all after current file1
            file2 = files[j]

            # if files have different extensions, break
            if file1.split('.')[-1] != file2.split('.')[-1]: 
                break

            # if file contents do not match, break
            with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
                if f1.read() != f2.read():
                    break

            # delete the first one, and break out of this loop
            #os.remove(file1)
            new_name = os.path.join('/home/chad/Desktop/to_delete',  f'{u.get()} {os.path.basename(file1)}')
            _ = shutil.move(file1, new_name)
            copies_found.append((file1, file2))
            break

for i in copies_found:
    print(i)


# part 4v2: i have 2 folders, a good library and old library. if a book in the old
# library is in the new library, we delete it. otherwise we check for copies as usual
# ============================================================================
# ============================================================================

