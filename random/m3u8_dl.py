
import selenium
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait

import requests
import m3u8

import time, re, os
os.chdir('/home/chad/Desktop/bastl/py')

from pprint import pprint as p

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

# logins
username = "pwnagecorp2@gmail.com"
password = "vladtchenko"

# initialize the driver
driver = webdriver.Firefox(executable_path = '/home/chad/Desktop/bastl/4path/geckodriver')

# head to login page
driver.get("https://www.reaktortutorials.com/signin")

# TODO: apparently the next stuff pops up only after I inspect it?

# perform login by typing info and clicking button
driver.find_element(by = webdriver.common.by.By.CLASS_NAME, value = "visited").send_keys(username)
driver.find_element(by = webdriver.common.by.By.NAME, value = 'signinPassword').send_keys(password)
driver.find_element(by = webdriver.common.by.By.NAME, value = 'signinButton').click()
driver.get("https://www.reaktortutorials.com/categories")


# get the categories, 39 of them
__category_urls = driver.find_elements(by = webdriver.common.by.By.CLASS_NAME, value = 'ps-authorlist-item')
__category_names = driver.find_elements(by = webdriver.common.by.By.CLASS_NAME, value = 'ps-authorlist-item-link')
__category_amounts = driver.find_elements(by = webdriver.common.by.By.CLASS_NAME, value = 'ps-authorlist-item-count')
_category_amounts = [int(i.get_attribute('innerHTML')) for i in __category_amounts]
_category_urls = [i.get_property('href') for i in __category_urls]
_category_names = [re.sub('[^0-9a-zA-Z_]+', '', i.get_property('innerHTML').replace(' ', '_'))[:20] for i in __category_names]
assert len(_category_urls) == 39, 'missing categories'
assert len(_category_amounts) == 39, 'missing amounts'
assert len(_category_names) == 39, 'missing names'
for i in range(39):
    print(f'{_category_amounts[i]}\t{_category_names[i]}\t{_category_urls[i][43:]}')



# FOR EACH CATEGORY
for _categ_idx in range(31,len(_category_urls)):

    _categ_name = str(_categ_idx).zfill(3) + '_' + _category_names[_categ_idx]
    _categ_amount = _category_amounts[_categ_idx] # number of videos we should get
    _categ_url = _category_urls[_categ_idx]

    # make directory for that category
    make_dir(_categ_name)
    os.chdir(_categ_name)

    # Open a new window and go to category
    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[1]) # Switch to the new window and open new URL
    driver.get(_categ_url)
    time.sleep(20) # wait for it to load

    # get list of videos 
    __vid_urls = driver.find_elements(by = webdriver.common.by.By.CLASS_NAME, value = 'ps-video-thumbnail-options-watch-now')
    __vid_names = driver.find_elements(by = webdriver.common.by.By.CLASS_NAME, value = 'ps-video-thumbnail-title')
    _vid_urls = [i.get_property('href') for i in __vid_urls]
    _vid_names = [re.sub('[^0-9a-zA-Z_]+', '', i.get_attribute('innerHTML').replace(' ', '_'))[:20] for i in __vid_names]
    assert len(_vid_urls) == _categ_amount, 'did not find enough video urls!'
    assert len(_vid_names) == _categ_amount, 'did not find enough video names!'

    # FOR EACH VIDEO 
    for _vid_idx in range(len(_vid_urls)):
        _vid_url = _vid_urls[_vid_idx]
        _vid_name = _vid_names[_vid_idx]
        FILE_NAME = str(_vid_idx).zfill(2) + f'_{_vid_name}.ts'

        print(f'starting video {_vid_idx}, category {_categ_idx}')

        # Open a new window and go to video
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[2]) 
        driver.get(_vid_url)
        time.sleep(20) # wait for it to load


        # check for file attachments, and if they exist, get em 
        _attachments = driver.find_elements(by = webdriver.common.by.By.CLASS_NAME, value = 'ps-video-attachment')
        if len(_attachments) > 0:
            print(f'category {_categ_idx} video {_vid_idx}')
            for __i in range(len(_attachments)):
                _attachment = _attachments[__i]
                _url = _attachment.get_property('href')
                print(_url)
                # get file
                __site_file = _attachment.get_attribute('innerHTML')
                _site_file = __site_file.split('(')[0].strip()
                FILE_NAME = f'cat_{_categ_idx}_vid_{_vid_idx}_file_{__i}__{_site_file}'
                with open(FILE_NAME, 'wb') as f:
                    r = requests.get(_url)
                    _ = f.write(r.content)



        if False:
            # get the master m3u8 file
            # https://stackoverflow.com/questions/49368983/access-list-of-files-requested-as-a-page-loads-using-selenium
            timings = driver.execute_script("return window.performance.getEntries();")
            GOT_MASTER_M3U8 = False
            for _i in timings:
                _name = _i['name']
                if _name.__contains__('m3u8') and _name.__contains__('stream.mux'):
                    master_m3u8_url = _name
                    GOT_MASTER_M3U8 = True
            assert GOT_MASTER_M3U8, 'did not get the master m3u8'

            # master m3u8 file
            r = requests.get(master_m3u8_url)
            m3u8_master = m3u8.loads(r.text)

            # different resolutions
            options = []
            for i in range(len(m3u8_master.data['playlists'])):
                options.append((i, int(m3u8_master.data['playlists'][i]['stream_info']['resolution'].split('x')[0])))
            idx = sorted(options, key = lambda x: x[1])[-1][0]

            # get list of m3u8 files 
            playlist_uri = m3u8_master.data['playlists'][idx]['uri']
            r = requests.get(playlist_uri)
            playlist = m3u8.loads(r.text)

            # get video
            with open(FILE_NAME, 'wb') as f:
                for i, segment in enumerate(playlist.data['segments']):
                    uri = segment['uri']
                    r = requests.get(uri)
                    _ = f.write(r.content)

        # close video tab
        driver.close()
        driver.switch_to.window(driver.window_handles[1])

    # close category tab
    driver.close()
    driver.switch_to.window(driver.window_handles[0])
    os.chdir('..')

os.getcwd()














# for category 30 you need to manually click on the other pages 
# since there are 3 pages.

# RUN THIS PART ONCE:
_categ_idx = 30
_categ_name = str(_categ_idx).zfill(3) + '_' + _category_names[_categ_idx]
_categ_amount = _category_amounts[_categ_idx] # number of videos we should get
_categ_url = _category_urls[_categ_idx]
make_dir(_categ_name)
os.chdir(_categ_name)
driver.execute_script("window.open('');")
driver.switch_to.window(driver.window_handles[1]) # Switch to the new window and open new URL
driver.get(_categ_url)
time.sleep(20) # wait for it to load


# RUN THIS PART, THEN CLICK NEXT PAGE, DO THIS 3 TIMES
__vid_urls = driver.find_elements(by = webdriver.common.by.By.CLASS_NAME, value = 'ps-video-thumbnail-options-watch-now')
__vid_names = driver.find_elements(by = webdriver.common.by.By.CLASS_NAME, value = 'ps-video-thumbnail-title')
_vid_urls = [i.get_property('href') for i in __vid_urls]
_vid_names = [re.sub('[^0-9a-zA-Z_]+', '', i.get_attribute('innerHTML').replace(' ', '_'))[:20] for i in __vid_names]
for _vid_idx in range(len(_vid_urls)):
    _vid_url = _vid_urls[_vid_idx]
    _vid_name = _vid_names[_vid_idx]
    FILE_NAME = str(47 + _vid_idx).zfill(2) + f'_{_vid_name}.ts'

    print(f'starting video {_vid_idx}, category {_categ_idx}')

    # Open a new window and go to video
    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[2]) 
    driver.get(_vid_url)
    time.sleep(20) # wait for it to load

    # check for file attachments, and if they exist, get em 
    _attachments = driver.find_elements(by = webdriver.common.by.By.CLASS_NAME, value = 'ps-video-attachment')
    if len(_attachments) > 0:
        print(f'category {_categ_idx} video {_vid_idx}')
        for __i in range(len(_attachments)):
            _attachment = _attachments[__i]
            _url = _attachment.get_property('href')
            print(_url)
            # get file
            __site_file = _attachment.get_attribute('innerHTML')
            _site_file = __site_file.split('(')[0].strip()
            FILE_NAME = f'cat_{_categ_idx}_vid_{_vid_idx}_file_{__i}__{_site_file}'
            with open(FILE_NAME, 'wb') as f:
                r = requests.get(_url)
                _ = f.write(r.content)

    if False:

        # get the master m3u8 file
        # https://stackoverflow.com/questions/49368983/access-list-of-files-requested-as-a-page-loads-using-selenium
        timings = driver.execute_script("return window.performance.getEntries();")
        GOT_MASTER_M3U8 = False
        for _i in timings:
            _name = _i['name']
            if _name.__contains__('m3u8') and _name.__contains__('stream.mux'):
                master_m3u8_url = _name
                GOT_MASTER_M3U8 = True
        assert GOT_MASTER_M3U8, 'did not get the master m3u8'

        # master m3u8 file
        r = requests.get(master_m3u8_url)
        m3u8_master = m3u8.loads(r.text)

        # different resolutions
        options = []
        for i in range(len(m3u8_master.data['playlists'])):
            options.append((i, int(m3u8_master.data['playlists'][i]['stream_info']['resolution'].split('x')[0])))
        idx = sorted(options, key = lambda x: x[1])[-1][0]

        # get list of m3u8 files 
        playlist_uri = m3u8_master.data['playlists'][idx]['uri']
        r = requests.get(playlist_uri)
        playlist = m3u8.loads(r.text)

        # get video
        with open(FILE_NAME, 'wb') as f:
            for i, segment in enumerate(playlist.data['segments']):
                uri = segment['uri']
                r = requests.get(uri)
                _ = f.write(r.content)

    # close video tab
    driver.close()
    driver.switch_to.window(driver.window_handles[1])




import time

while True:
    print(1)
    time.sleep(10)







































































# for each category:
# 1. check they have good resolution
# 2. check the number of videos matches the website

from subprocess import Popen, PIPE
import glob
soy_dir = '/home/chad/Desktop/bastl/py'
soy_folders = sorted([i for i in glob.glob(os.path.join(soy_dir, '*')) if os.path.isdir(i)])


# 1. check the video resolution using ffmpeg. sometimes it yields Null for a good video
counter = 0
for _categ_idx in range(39):
    soy_files = glob.glob(os.path.join(soy_folders[_categ_idx], '*'))
    for i in soy_files:
        cmd = "ffmpeg -i %s 2>&1 | grep Video: | grep -Po '\d{3,5}x\d{3,5}'" % i
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        di = p.communicate()
        print(di[0].decode().strip(), i)
    print(len(soy_files))
    counter += len(soy_files)

# 2. get the folders which don't match video_amount
for _i in range(39):
    soy_files = glob.glob(os.path.join(soy_folders[_i], '*'))
    if len(soy_files) != _category_amounts[_i]:
        print(_i, len(soy_files), _category_amounts[_i])

