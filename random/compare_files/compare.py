import os

cur_dir = os.getcwd()

_files = os.listdir()
_files.remove('compare.py')
_files = [os.path.join(cur_dir, i) for i in _files]
_files


_data = []
for i in _files:
    with open(i, 'rb') as _file:
        _data.append(_file.read())

_data[0] == _data[1]

len(_data[0])
len(_data[1])
