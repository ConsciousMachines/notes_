

# # h1..h4 : titles, process them separately 
# # a: anchor, hyperlink. 
# # dt: description term. like a header
# # th: table header
# # code: code
# # span: idk what it does
# # b: bold
# # div: container for a style or something. do recursively
# # pre: preformatted text, in this case code snippets. only comments need translating 
# # table / tr / td: table, row, data
# # ol / ul / dl : ordered unordered description list 
# # blockquote: makes it a quote
# # nav: table of contents
# book 2 tags:
# strong, i, sub, sup
# also added img, hr, br because apparently they have text/tail sometimes 


# TODO: some weird html result from google translate such as '&amp;#39;' and '&amp;quot;' need to be adjusted manually
#       idea: go over the translation patch files and replace things that start with "&"
#       .replace('&amp;quot;',"'").replace('&amp;#39;',"'")


import os
import re
import time
import zipfile 
import hashlib
from lxml import etree
from google.cloud import translate_v2 as translate
# make sure the private key is in the environment, for google API to work (put it in .bashrc)
# export GOOGLE_APPLICATION_CREDENTIALS="/home/chad/Downloads/symmetric-span-382608-57553f98cfda.json"


class text_processor:
    def __init__(self):
        
        self.counter                = 0

        self.translate_client       = translate.Client()
        self.japanese_re            = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')

        self.DEBUG                  = False
        # self.DEBUG                  = True

    # function that sends a string to Google Cloud Translate API. assumes input is Japanese text.
    def google_translate(self, text):

        text_s                      = text.strip()              # remove spaces
        spaces                      = text.split(text_s)        # save the spaces so we can restore them later
        
        if self.DEBUG == True:
            text_ret                = f'yes japanese: {text}'
        if self.DEBUG == False:
            time.sleep(1)                                    # wait 
            translation             = self.translate_client.translate(text_s, source_language='ja', target_language='en')
            text_tr                 = translation['translatedText']
            text_ret                = spaces[0] + text_tr + spaces[1]

            print(text_s, text_ret.strip(), '', sep='\n')

        return text_ret

    # replace a chunk of Japanese text with a guid [[12345]], and write that guid with translation to patch file
    # this way i can store the translation separately and then patch files
    def translate_text(self, text):
        if self.japanese_re.search(text):            # if text contains Japanese

            # include hash of original text so we know we are replacing the correct translation, file order is correct
            hash_object             = hashlib.sha256(text.encode('utf-8'))
            hex_dig                 = hash_object.hexdigest()

            # generate guid for a chunk of text. replace text with guid. write (guid,hash,translation) to separate file.
            replace_text            = f'\n\n[[[{str(self.counter).rjust(20,"0")}---{hex_dig}]]]'
            translated_text         = self.google_translate(text)
            self.counter            = self.counter + 1
            self.patch_file_ptr.write(replace_text)
            self.patch_file_ptr.write(translated_text)
            return replace_text
        else:
            return text # otherwise return original text
    
    def translate_init(self, patch_file_ptr): # we are given opened file to store translations in 
        self.patch_file_ptr         = patch_file_ptr

    def patch_text(self, text):
        if self.japanese_re.search(text):

            # get a hash so we can make sure we have the correct translated chunk
            hash_object             = hashlib.sha256(text.encode('utf-8'))
            _hash                   = hash_object.hexdigest()

            # the guid / index relies on Python listing the zipped files in the same order each time
            _patch                  = self.chunks[self.counter]
            self.counter            = self.counter + 1

            assert _patch[0] == _hash, 'hash not matched. are the files the same?'
            return _patch[1] # translated text
        else:
            return text # text which does not need patching

    def patch_init(self, patch_file_path):
        self.counter                = 0

        # load the translations from the patch file
        with open(patch_file_path, 'r', encoding='utf-8') as f:
            patch                   = f.read()
            patch                   = patch.split('\n\n[[[')[1:] # the first element is '' due to split() so we skip it 

        self.chunks = {}
        for chunk in patch:
            parts                   = chunk.split(']]]')
            assert len(parts) == 2, f'more than 2 parts for some reason: {chunk}'

            _guid, _hash            = parts[0].split('---')
            self.chunks[int(_guid)] = (_hash, parts[1]) # used guid as index, hash to double check


# remove the text of the element and its childs
def remove_text(elem):
    elem.text       = ''
    elem.tail       = ''

    for child in elem:
        remove_text(child)
        

# remove namespace from tag. 
# example: {http://www.w3.org/1999/xhtml}head -> head
def strip_tag_namespace(long_tag): 
    return long_tag.split('}')[-1]


# for a paragraph, combine all text together because that way it gets translated better with context
def process_paragraph(elem, processing_func):

    # get all the text inside paragraph (except root tail)
    text            = ''.join(list(elem.itertext())) 
    elem.text       = processing_func(text)
    
    # empty the childs' text/tail, since we already got it above
    for child in elem.getchildren():
        remove_text(child)
    
    # the <p> tail should be just a newline
    if elem.tail:
        assert elem.tail.strip() == '', f'paragraph tail is not whitespace: {repr(elem.tail)}'


# call a different processing function depending on element's tag (either process text together or separately)
def walk_element(elem, processing_func):
    
    if type(elem.tag) != str:
        return # apparently this is only for comments, and its type is a cython function

    _tag = strip_tag_namespace(elem.tag)

    # paragraph: concatenate everything together, contains full sentences cut up and distributed among child elements
    # dd / li: description data / list item also a paragraph 
    if _tag in ['p', 'dd', 'li']:
        process_paragraph(elem, processing_func)

    # things that are not full sentences: translate .text and .tail separately. do not combine them
    elif _tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'a', 'dt', 'th', 'code', 'span', 'b', 'div', 'pre', 'table', 'tr', 'td', 'ol','ul', 'dl', 'blockquote', 'nav', 'body', 'strong', 'i', 'sub', 'sup', 'img', 'hr', 'br']:

        # .text attribute holds the text content that appears immediately after the opening tag of an element.
        if elem.text:
            elem.text = processing_func(elem.text) # process it separately
        
        # When an element has child nodes, the .text attribute contains the text content between 
        # the opening tag of the element and the opening tag of the first child element. 
        for child in elem.getchildren():
            walk_element(child, processing_func) # this will consider the tag of the childs and if they are <p>, process it as paragraph
        
        # .tail attribute contains the text content between the closing tag of the element and the opening tag of the next sibling element.
        if elem.tail:
            elem.tail = processing_func(elem.tail) # process it separately

    else:
        assert False, f'unsupported tag: {_tag}'


# part 1: create a translation patch in the directory where the book is
def create_translation_patch(book_path):

    book_name          = os.path.basename(book_path).replace('.epub', '')
    patch_file_path    = os.path.join(os.path.dirname(book_path), f'{book_name}_patch.txt')
    tp                 = text_processor()
        
    with zipfile.ZipFile(book_path, 'r') as zip_file:

        with open(patch_file_path, 'w', encoding='utf-8') as patch_file_ptr:
            
            tp.translate_init(patch_file_ptr)
            
            for file_name in zip_file.namelist():

                if os.path.splitext(file_name)[1] == '.xhtml': # chapters are .xhtml files
                    print(file_name)

                    with zip_file.open(file_name, 'r') as f:
                        chapter_content = f.read() # bytes, pass to etree.fromstring() which can read the utf-8 declaration in header

                        parser         = etree.XMLParser(recover=True)
                        root           = etree.fromstring(chapter_content, parser)

                        # step 1. the root has 2 elements: head and body. skip head.
                        root_childs    = [strip_tag_namespace(i.tag) for i in root.getchildren()]
                        assert root_childs == ['head', 'body'], 'root has other nodes aside from head/body'
                        body           = root.getchildren()[1]

                        # step 2.  body is list of headers/paragraphs/etc as they appear in the render.
                        # we write a translating function for each type of node, for example header vs paragraph
                        walk_element(body, processing_func=tp.translate_text)
    
    return patch_file_path


# part 2: patch a book given a patch file
def patch_book(book_path, patch_file_path):

    book_name                          = os.path.basename(book_path).replace('.epub', '')
    tp                                 = text_processor()
        
    with zipfile.ZipFile(book_path, 'r') as old_zip: # original book

        with zipfile.ZipFile(book_path.replace(book_name, book_name + '_patch'), 'w') as new_zip: # new patched book

            tp.patch_init(patch_file_path)

            for file_name in old_zip.namelist(): # loop over files in the book

                with old_zip.open(file_name, 'r') as f:
                    file_content       = f.read() # bytes, pass to etree.fromstring() which can read the utf-8 declaration in header

                    if os.path.splitext(file_name)[1] == '.xhtml': # chapters are .xhtml files
                        print(file_name)

                        parser         = etree.XMLParser(recover=True)
                        root           = etree.fromstring(file_content, parser)

                        # step 1. the root has 2 elements: head and body. skip head.
                        root_childs    = [strip_tag_namespace(i.tag) for i in root.getchildren()]
                        assert root_childs == ['head', 'body'], 'root has other nodes aside from head/body'
                        body           = root.getchildren()[1]

                        # step 2.  body is list of headers/paragraphs/etc as they appear in the render.
                        # we write a translating function for each type of node, for example header vs paragraph
                        walk_element(body, processing_func=tp.patch_text)

                        # get the doctype declaration and other stuff that gets lost in the parsing
                        preamble       = file_content.decode('utf-8').split('<html ')[0] # this is a utf-8 string

                        # convert tree back to xhtml string
                        output_xhtml   = etree.tostring(root, pretty_print=True, encoding='utf-8', xml_declaration=False, doctype='').decode('utf-8')
                        file_content   = preamble + output_xhtml

                    # write content to patched book
                    new_zip.writestr(file_name, file_content)


book_path          = r'/home/chad/Desktop/DL1.epub'
patch_file_path    = create_translation_patch(book_path)
patch_book(book_path, patch_file_path)
