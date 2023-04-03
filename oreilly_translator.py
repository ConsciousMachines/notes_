
from google.cloud import translate_v2 as translate
# make sure the private key is in the environment, for google API to work:
# export GOOGLE_APPLICATION_CREDENTIALS="/home/chad/Downloads/symmetric-span-382608-91d2bcbfba3e.json"

import os
import re
import time
import xml.etree.ElementTree as ET


# Create a client object for the Translation API
translate_client = translate.Client() 
# Define the source language and target language
source_language = 'ja' 
target_language = 'en'
# Compile a regular expression to match Japanese characters
japanese_re = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')
# count number of chars sent to API
c = counter()


# a simple counter to count the characters processed
class counter:
    def __init__(self):
        self.count = 0
    def add(self, n):
        self.count += n


# function that sends a string to Google Cloud Translate API
def translate_text(text):
    if japanese_re.search(text):       # if text contains japanese 

        text_s = text.strip()          # remove spaces
        spaces = text.split(text_s)    # the spaces so we can restore them

        if text_s != '':               # if there are characters
            c.add(len(text_s))         # keep count of how many chars we're processing

            time.sleep(1)              # wait 
            translation = translate_client.translate(text_s, source_language=source_language, target_language=target_language)
            text_tr = translation['translatedText']
            text_ret = spaces[0] + text_tr + spaces[1]
            print(text.strip())
            print(text_ret.strip())
            print()

            return text_ret
    return text                        # otherwise return the original text


# Recursively copy an XML element without its text content
def copy_elem(elem):
    new_elem = ET.Element(elem.tag, attrib=elem.attrib)
    new_elem.text = None
    new_elem.tail = None
    for child in elem:
        new_child = copy_elem(child)
        new_elem.append(new_child)
    return new_elem


def walk_and_process(elem):

    # Create a new element with the same tag and attributes
    new_elem = ET.Element(elem.tag, attrib = elem.attrib)

    # check the actual tag without namespace. 
    tag_without_ns = elem.tag.split('}')[-1]

    # title element is never rendered, skip it
    if tag_without_ns == 'title':
        new_elem.text = None

    # if tag is p/li/dd, process it as a paragraph
    elif tag_without_ns in ['p', 'li', 'dd']:
        
        # take out text recursively 
        all_text = ''.join(list(elem.itertext()))

        # translate
        new_elem.text = translate_text(all_text)

        # Loop over the children and copy them without text
        for child in elem:

            # Copy and add the new child to the new element
            new_elem.append(copy_elem(child))

    # otherwise, process elem as a tiny blob of text
    else: 

        # Loop over the children and process each one
        for child in elem:

            # Recursively process the child
            new_child = walk_and_process(child)

            # Add the new child to the new element
            new_elem.append(new_child)

        # translate the main text
        if elem.text is not None:
            new_elem.text = translate_text(elem.text)

        # translate the tail text
        if elem.tail is not None:
            new_elem.tail = translate_text(elem.tail)

    return new_elem


chaps_dir = r'/home/chad/Desktop/DL3/OEBPS/'
file_dirs = [os.path.join(chaps_dir,i) for i in os.listdir(chaps_dir) if i.split('.')[-1] == 'xhtml']

for file_dir in file_dirs:
    tree = ET.parse(file_dir)
    root = tree.getroot()
    new_root = walk_and_process(root)

    final = ET.tostring(new_root,encoding='utf-8').decode().replace('&amp;quot;',"'").replace('&amp;#39;',"'")

    with open(file_dir, "w", encoding = 'utf-8') as f:
        _ = f.write(final)


# NOTES
# looking at the structure of the book, Japanese is usually in small blobs like comments 
#   or figure descriptions, or it constitutes a big paragraph. The small blobs get translated
#   individually. For paragraphs, the text inside the XML file is partitioned by elements like
#   inline code or LaTeX. The translation is much better if we conglomerate all the text together,
#   but that destroys the order of things. So we get a good translation, and then everything in 
#   the paragraph will be to the right of it. 
# the parsing seems to modify some things like remove unnecessary namespace declarations.
# things that should be done as a whole paragraph together: <p>, <li>, <dd>
#   (i looked through the whole book)
# <code> segments will be translated as i translate things now (just the text of one element)





# # function to copy text to clipboard (linux)
# import subprocess
# def copy_to_clipboard(text):
#     p = subprocess.Popen(['xsel', '-bi'], stdin=subprocess.PIPE)
#     p.communicate(input=text.encode())


# def test_1():
        
#     # WORKING PART 1: walk the tree in the same was as it is displayed, and as itertext().
#     def walk_elem(elem):
#         # Yield the text of this element, if any
#         if elem.text:
#             yield elem.text

#         # Recursively yield the text of each child element
#         for child in elem:
#             yield from walk_elem(child)

#             # Yield the tail of this child element, if any
#             if child.tail:
#                 yield child.tail

#     tree = ET.parse(file_dir)
#     root = tree.getroot()
#     walked_text = ''.join(list(walk_elem(root)))
#     real = ''.join(list(root.itertext()))
#     walked_text == real

#     # triple check
#     with open(file_dir, 'r', encoding='utf-8') as f:
#         chap_content = f.read()
#     japanese_sentences = re.sub(r'<[^>]+>', '\n', chap_content).replace('\n','')
#     # print(japanese_sentences)

#     return (walked_text == real) and (japanese_sentences == real.replace('\n',''))


# def test_2():
        
#     # WORKING PART 2: WALK THE TREE, CHANGE TEXT, AND RETURN A SECOND TREE
#     def walk_and_process(elem):
#         # Create a new element with the same tag
#         new_elem = ET.Element(elem.tag)

#         # Copy over the attributes
#         new_elem.attrib.update(elem.attrib)

#         # Loop over the children and process each one
#         for child in elem:
#             # Recursively process the child
#             new_child = walk_and_process(child)

#             # Add the new child to the new element
#             new_elem.append(new_child)

#         # Convert the text to upper case and add it to the new element
#         if elem.text is not None:
#             new_elem.text = elem.text

#         # Convert the tail to upper case and add it to the new element
#         if elem.tail is not None:
#             new_elem.tail = elem.tail

#         return new_elem

#     tree = ET.parse(file_dir)
#     root = tree.getroot()
#     new_root = walk_and_process(root)

#     r1 = ET.tostring(root,encoding='utf8').decode()
#     r2 = ET.tostring(new_root,encoding='utf8').decode()
#     # print(r1)
#     return r1 == r2 


# assert test_1()
# assert test_2()


# Define the text to translate
# text = '日'

# Translate the text from Japanese to English
# translation = translate_client.translate(text, source_language=source_language, target_language=target_language)
# translation
# translation['translatedText']

# Print the translated text
# print(translation['input'])
# print(translation['translatedText'])


# Text can also be a sequence of strings, in which case this method
# will return a sequence of results for each text.
# result = translate_client.translate(text, target_language=target)
# result

# https://cloud.google.com/translate/docs/languages
# https://cloud.google.com/translate/docs/basic/discovering-supported-languages

