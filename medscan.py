from PIL import Image
import cv2 
import numpy as np
from scipy.ndimage import interpolation as inter
import pytesseract
import re
from more_itertools import intersperse
from ms_keywords import sections2kws, fields2kws # {str: [str]}
from collections import defaultdict
from functools import reduce

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)[1] # was THRESH_BINARY INITIALLY
    
#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((3,3),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

def correct_skew(image, delta=1, limit=5, input_is_gray=False):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score
    if input_is_gray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray=image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)
    
    return rotated
    return best_angle, rotated

def prettier_text(input_text):
    cleaner = lambda match: '\n' if '\n' in match.group() else ' '
    result = re.sub(r'\s+', cleaner, input_text)
    return result

def transpose_dict(input_dict):
    new_dict = {}
    for group, elements in input_dict.items():
        new_dict.update(dict.fromkeys(elements,group))
    return new_dict

kws2sections = transpose_dict(sections2kws)

def chinchoppa(text):
     
    fields = {}
    
    # name
    for kw in fields2kws['name']:
        temp = text.split(kw, 1)
        if len(temp)==1:
            continue
        temp = temp[1]
        name_match = re.search('\w', temp) 
        if name_match:
            name = ' '.join(temp[name_match.start():].split(' ')[:3])
            fields['name'] = name
            break

    # TODO date 
 
    sections_kws = kws2sections.keys()
    sections_kws = sorted(sections_kws, key=len, reverse=True)
    # break down text into keyword and surrounding text until we run out of keywords
    # keep text in original order in pieces, and its type in is_kw
    pieces = [text]
    is_kw = [False]
    for kw in sections_kws:
        piece_idx = 0
        while piece_idx < len(pieces):
            if not is_kw[piece_idx]:
                kw_ = kw[0].upper() + kw[1:]
                sub_pieces = re.split(f'\n{kw_}|{kw_}[\n:]', pieces[piece_idx]) # с большой буквы, заканчивается на двоеточие
                # remove non-alphanumeric symbols in the start of the string 
                for i in range(1, len(sub_pieces)):
                    match = re.search(f'[\w]', sub_pieces[i]) 
                    if match:
                        sub_pieces[i] = sub_pieces[i][match.start():]
                sub_is_kw = [False]*len(sub_pieces)
                sub_is_kw = list(intersperse(True, sub_is_kw))
                sub_pieces = list(intersperse(kw, sub_pieces))
                pieces[piece_idx:piece_idx+1] = sub_pieces
                is_kw[piece_idx:piece_idx+1] = sub_is_kw
                piece_idx += len(sub_pieces) 
            else:
                piece_idx += 1
    
    # concat consecutive blocks of non header text 
    piece_idx = 1
    while piece_idx < len(pieces):
        if not is_kw[piece_idx] and not is_kw[piece_idx-1]:
            is_kw.pop(piece_idx)
            pieces[piece_idx-1] += pieces[piece_idx]
            pieces.pop(piece_idx)
        else:
            piece_idx += 1
    # now it's always ...header-nonheader-header-...
    
    # make sure we have in pieces n pairs of (header, text)
    if is_kw[0] == False:
        pieces.insert(0, 'junk')
        is_kw.insert(0, True)
    if len(pieces) % 2 != 0:
        pieces.append('')
    
    # finally put them in our dict 
    kw2sectiontext = {}   
    for piece_idx in range(0,len(pieces),2):
        kw2sectiontext[pieces[piece_idx]] = pieces[piece_idx+1]
    
    # move from keywords to internal section names
    result = kwdict2sectiondict(kw2sectiontext)

    for section in result:
        result[section] = prettier_text(result[section]) # prettify them
    result.update(fields) # add fields to the result (such as name and date)
    return result

def dict2debug(dict_,start='<br/>---',end='---<br/>'):
    result = ''
    for kw in dict_:
        if kw=='junk':
            result+=dict_['junk']
        else:
            result+=start+kw.upper()+end+dict_[kw]
    return result

def kwdict2sectiondict(kw2sectiontext):
    # {keyword : value} - > {section : value}
    sectiondict = defaultdict(str)
    for kw in kw2sectiontext:
        section = kws2sections[kw]
        sectiondict[section] += kw2sectiontext[kw] + '\n'
    return sectiondict
 
preprocessing_functions = [get_grayscale,         correct_skew,]
apply_preprocessing = lambda input_img: reduce(lambda img, func: func(img), preprocessing_functions, input_img)

def predict(input_image_list, debug=False):
    preprocessed_images = [apply_preprocessing(img) for img in input_image_list]
    raw_text = ' '.join([pytesseract.image_to_string(img, lang='rus+eng',) for img in preprocessed_images])
    result = chinchoppa(raw_text)
    if debug:
        return dict2debug(result)
    else:
        return result
