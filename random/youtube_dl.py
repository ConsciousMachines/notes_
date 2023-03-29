# script to run "youtube-dl" in a loop because it keeps stopping when it gets a random 403 error


import subprocess 
import os


num_vids = 83
dirr = '/home/chad/Desktop/media/youtube/mewsic_2021'
playlist = 'https://www.youtube.com/playlist?list=PL-pRcnfikSGfWep3Jqh5wLISAqbKINM13'


def run(args): # https://stackoverflow.com/questions/54091396/live-output-stream-from-python-subprocess
  with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
    for line in process.stdout:
      print(line.decode('utf8'))


def make_dir(dirr):
  if not os.path.exists(dirr):
    os.mkdir(dirr)


command = f'''youtube-dl {playlist} --extract-audio --audio-format mp3 --audio-quality 0 --verbose --download-archive downloaded.txt'''.split(' ')
make_dir(dirr)
os.chdir(dirr)
while len([i for i in os.listdir(dirr) if i.split('.')[-1] == 'mp3']) != num_vids:
    run(command)



run(command)
