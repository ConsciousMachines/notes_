

import os 


class prompt_table: # basically a set() that can be indexed
    def __init__(self):
        # it has a list of words
        self.words = [] 
    def add(self, word):
        # if our list has the word, return its index
        if self.words.__contains__(word):
            return self.words.index(word)
        # otherwise add the word and return its index
        else:
            self.words.append(word)
            return len(self.words) - 1
    def get(self, idx):
        # retrieve a word knowing its index
        my_assert(idx < len(self.words), 'requested index out of range')
        return self.words[idx]


def my_assert(condition, error_msg):
    # assert a condition and exit if its true,  so i know it triggered
    if (condition) == False:
        print(error_msg)
        exit()


def split_prompt(prompt):
    # take the full text prompt and return a list, split by comma
    prompt = prompt.replace('\\n','')
    parts = [i.strip() for i in prompt.split(',')]
    while '' in parts:
        parts.remove('')
    return parts


def split_technicals(_tec):

    # check that we dont deal with img2img
    if _tec.__contains__('Denoising strength') or _tec.__contains__('Mask blur'):
        my_assert(True, 'this is an img2img')
    _tec_parts = [i.strip() for i in _tec.split(',')]

    # steps | sampler | cfg | seed | size | hash | model
    _step_str, _step_val = [i.strip() for i in _tec_parts[0].split(':')]
    _samp_str, _samp_val = [i.strip() for i in _tec_parts[1].split(':')]
    _cfgs_str, _cfgs_val = [i.strip() for i in _tec_parts[2].split(':')]
    _seed_str, _seed_val = [i.strip() for i in _tec_parts[3].split(':')]
    _size_str, _size_val = [i.strip() for i in _tec_parts[4].split(':')]
    _hash_str, _hash_val = [i.strip() for i in _tec_parts[5].split(':')]
    # forget the model, its right before the binary blob and causes errors. just use hash
    #_modl_str, _modl_val = [i.strip() for i in _tec_parts[6].split(':')]

    # assert that we parsed proper attributes
    my_assert([_step_str, _samp_str, _cfgs_str, _seed_str, _size_str, _hash_str] == ['Steps', 'Sampler', 'CFG scale', 'Seed', 'Size', 'Model hash'],'did not get the same technical details (maybe its img2img?)')
    _size_1, _size_2 = _size_val.split('x')

    # make the actual list of values 
    parts = [_step_val, _samp_val, _cfgs_val, _seed_val, _size_1, _size_2, _hash_val]
    return parts


def extract_prompt(_file):

    # file structure (?):
    # 1. it starts with 'parameters\\x00'
    # 2. the last ascii is followed by random bytes, then a \\x00

    f = open(_file, 'rb')
    _bin = f.read() # the entire file binary
    _bin1 = _bin[:10000] # subset of it (ASSUMPTION: metadata is <10k chars)
    f.close()

    _bin2 = str(_bin1) # convert it to string to cut stuff out 

    _idx1 = _bin2.find('parameters\\x00') # find where the metadata chunk starts
    my_assert(_idx1 != -1, 'did not find parameters (prob bad binary digit after it?)')

    _idx2 = _bin2[_idx1:].find('Negative prompt:') # negative prompt starts
    my_assert(_idx2 != -1, 'did not find Negative Prompt')
    _idx2 += _idx1 # since we offset 

    _idx3 = _bin2[_idx2:].find('Steps:') # technical details start
    my_assert(_idx3 != -1, 'did not find Steps')
    _idx3 += _idx2 # since we offset

    _idx4 = _bin2[_idx3:].find(', Model:') # we shall use 'Model:' as the end of our technicals
    my_assert(_idx4 != -1, 'did not find Model:')
    _idx4 += _idx3 # since we offset

    _poz = _bin2[_idx1 + len('parameters\\x00'):_idx2]
    _poz_words = split_prompt(_poz)

    _neg = _bin2[_idx2 + len('Negative prompt:'):_idx3]
    _neg_words = split_prompt(_neg)

    _tec = _bin2[_idx3:_idx4]
    _tec_words = split_technicals(_tec)

    return _poz_words, _neg_words, _tec_words


def convert_file_to_prompt_indices(_file, pt):

    # read file meta data, to extract poz/neg/technical prompt
    _poz_words, _neg_words, _tec_words = extract_prompt(_file)

    # convert the words into indices into the prompt table
    _poz_idx = [pt.add(i) for i in _poz_words]
    _neg_idx = [pt.add(i) for i in _neg_words]
    #_tec_idx = [pt.add(i) for i in _tec_words]

    # test to see if it even works
    _poz_test = [pt.get(i) for i in _poz_idx]
    _neg_test = [pt.get(i) for i in _neg_idx]
    #_tec_test = [pt.get(i) for i in _tec_idx]
    my_assert(_poz_words == _poz_test, 'failed poz test')
    my_assert(_neg_words == _neg_test, 'failed neg test')
    #my_assert(_tec_words == _tec_test, 'failed tec test')

    return _poz_idx, _neg_idx, _tec_words


#######################################################################################
#######################################################################################


_sd_dir = r'/home/chad/Desktop'
_all_files = [os.path.join(_sd_dir,i) for i in os.listdir(_sd_dir) if i.split('.')[-1] == 'png']
len(_all_files)

for _file in _all_files:
    _poz_words, _neg_words, _tec_words = extract_prompt(_file)
    print()
    print(','.join(_poz_words))
    print(','.join(_neg_words))
    print(','.join(_tec_words))


#######################################################################################
#######################################################################################









# get files
_sd_dir = r'/home/chad/Desktop/diffusion'
_exclude = ['_img2img']
_dirs = [os.path.join(_sd_dir,i) for i in os.listdir(_sd_dir) if i not in _exclude]
_dirs = [i for i in _dirs if os.path.isdir(i)]
_all_files = [os.path.join(_dir, i) for _dir in _dirs for i in os.listdir(_dir) ]
len(_all_files)

# new prompt table
pt = prompt_table()


# step 1. go thru all files to collect the words
for _file in _all_files:
    _poz_idx, _neg_idx, _tec_words = convert_file_to_prompt_indices(_file, pt)

# step 2. shuffle the words
import random
random.shuffle(pt.words)

# step 3. go thru images again and get the indices, save them
imgs = {}
for _file in _all_files:
    _poz_idx, _neg_idx, _tec_idx = convert_file_to_prompt_indices(_file, pt)
    imgs[_file] = (_poz_idx, _neg_idx, _tec_words)

# step 4. encrypt the table 

from cryptography.fernet import Fernet
import numpy as np

pt.words
_msg = ';'.join(pt.words)
_b = bytes(_msg, 'ascii')
_orig = _b

# letters used in keys
_letters = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']



NN = 10
np.random.seed(2**32-1) # TODO: this is only 32 bits :(
keys = []
for i in range(NN):
    key = np.random.choice(_letters, 43) 
    key = bytes(''.join(key) + '=', encoding='ascii')
    keys.append(key)
    #key = Fernet.generate_key()
    #print('\t', key2, '\n\t', key)

for i in range(NN):
    f = Fernet(keys[i])
    _b = f.encrypt(_b)

_big_boi = _b

for i in range(NN):
    f = Fernet(keys[NN-i-1])
    _b = f.decrypt(_b)

_b == _orig

len(_big_boi)
_big_boi
rint = np.random.randint(2**63)
rint 










#######################################################################################
#######################################################################################


#######################################################################################
#######################################################################################





















# PROMT DUPLICATE REMOVER
prompts = '''

Mutation, extra limbs, extra arms, extra legs, bad anatomy
Mutation, lowres, {{bad anatomy}}, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, censored, {{{extra fingers}}}, {{{poor fingers}}}, {{{weird finger}}}, missing fingers, crossed eyes, fat, {{too big eyes}}, {{thick lips}}, cartoon, flat color, monochrome, greyscale, realistic_style
Mutation, lowres, {{bad anatomy}}, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, censored, {{{extra fingers}}}, {{{poor fingers}}}, {{{weird finger}}}, missing fingers, crossed eyes, fat, weird eyes, {{thick lips}},{{multiple view}}, long face, long body, bad face anatomy, cartoon, big eyes, {{{{{realistic_style}}}}}, fat, {{{girl in water}}}, underwater, {{girl in lake}}, only face, swimming,too much face focus, mature female, too much eye focus
Mutation, lowres, {{bad anatomy}}, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, censored, {{{extra fingers}}}, {{{poor fingers}}}, {{{weird finger}}}, missing fingers, crossed eyes, fat, realistic style, loli, {{too big eyes}}, {{thick lips}}, cartoon, rainbow
Mutation, lowres, {{bad anatomy}}, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, censored, {{{extra fingers}}}, {{{poor fingers}}}, {{{weird finger}}}, missing fingers, crossed eyes, fat, realistic style, loli, {{too big eyes}}, {{thick lips}}, cartoon, rainbow
Mutation, lowres, {{bad anatomy}}, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, censored, {{{extra fingers}}}, {{{poor fingers}}}, {{{weird finger}}}, missing fingers, crossed eyes, {{fat}}, weird eyes, {{thick lips}},{{multiple view}}, long face, long body, bad face anatomy, cartoon, too big eyes, unreal engine style, realistic face, flat color, cat ears, too big head, too long legs, too long waist
lowres, ((bad anatomy)), bad hands, bad face, text, error, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, extra fingers, poor fingers, weird finger, missing fingers, crossed eyes, fat, thick lips

'''
prompts = set(prompts.replace('\n', ',').replace(' ,', ',').replace(', ', ',').replace('{','').replace('}','').replace('[','').replace(']','').lower().split(','))
','.join(list(prompts))
for i in prompts:
    print(i,sep=',')



