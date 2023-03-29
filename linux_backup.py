import os
import shutil
import subprocess


# at this point the '_backups' folder should exist and contain stuff that doesn't move, 
# like Rack Rack2 techno_patch.vcv


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def copy_dir(_from_dir, _to_dir):
    assert 0 == subprocess.call(['cp', '-r', _from_dir, _to_dir]), f'FAILED: {_from_dir}'


def mov_dir(_from_dir, _to_dir):
    assert 0 == subprocess.call(['mv', _from_dir, _to_dir]), f'FAILED: {_from_dir}'


def movbak_dir(_to, _from):
    mov_dir(_from, _to)


_bakdir = '/home/chad/Desktop/_backups'

# copy preference folders (cant move or theyll break current running app)
copy_dir('/home/chad/.vscode-oss',             _bakdir)                         # codium prefs
copy_dir('/home/chad/.librewolf',              _bakdir)                         # librewolf prefs
copy_dir('/home/chad/.mozilla',                _bakdir)                         # firefox prefs

# move desktop items to _backups, then zip it for saving, then move back
#mov_dir('/home/chad/Bitwig Studio',            _bakdir)                         # bitwig projects
#mov_dir('/home/chad/hello',                    _bakdir)                         # Sedgewick Java
#mov_dir('/home/chad/.Rack',                    os.path.join(_bakdir, '_Rack'))  # VCV
#mov_dir('/home/chad/.Rack2',                   os.path.join(_bakdir, '_Rack'))
mov_dir('/home/chad/Desktop/2023.txt', _bakdir)                                  # notes
mov_dir('/home/chad/Desktop/skool', _bakdir)                                     # skool


# ZIP EVERYTHING UP FOR EXPORT 
shutil.make_archive('/home/chad/Desktop/_backups', 'zip', '/home/chad/Desktop/_backups')

# move the dirs back
movbak_dir('/home/chad/Desktop', os.path.join(_bakdir, 'skool'))                   # skool
movbak_dir('/home/chad/Desktop',       os.path.join(_bakdir, '2023.txt'))          # notes
#movbak_dir('/home/chad',               os.path.join(_bakdir, 'hello'))            # Sedgewick Java
#movbak_dir('/home/chad',               os.path.join(_bakdir, 'Bitwig Studio'))    # bitwig projects
#movbak_dir('/home/chad/.Rack',         os.path.join(_bakdir, '_Rack/.Rack'))      # VCV
#movbak_dir('/home/chad/.Rack2',        os.path.join(_bakdir, '_Rack/.Rack2'))

