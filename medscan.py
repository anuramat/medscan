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
    output_text = input_text
    output_text = ' '.join([i for i in output_text.split(' ') if i]) # remove consecutive spaces
    
    # remove spaces near line breaks
    output_text = list(output_text)
    for i in range(1, len(output_text)):
        if output_text[i] == '\n' and output_text[i-1] == ' ':
            output_text[i-1] = '' #
    output_text = ''.join(output_text)
    
    # remove consecutive line breaks
    output_text = '\n'.join([i for i in output_text.split('\n') if i]) # remove consecutive spaces
        
    return output_text

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
    pieces = [text]
    is_header = [False]
    for kw in sections_kws:
        piece_idx = 0
        while piece_idx < len(pieces):
            if not is_header[piece_idx]:
                kw_ = kw[0].upper() + kw[1:]
                temp = re.split(f'\n{kw_}|{kw_}[\n:]', pieces[piece_idx])#, flags=re.IGNORECASE) # с большой буквы, заканчивается на двоеточие
                # clean shit
                # убирает в начале блока хуевые символы (не альфанумерик)
                for i in range(len(temp)):
                    xd = re.search(f'[\w]', temp[i]) 
                    if xd:
                        temp[i] = temp[i][xd.start():]
                header_bullshit = [False]*len(temp)
                header_bullshit = list(intersperse(True, header_bullshit))
                temp = list(intersperse(kw, temp))
                pieces[piece_idx:piece_idx+1] = temp
                is_header[piece_idx:piece_idx+1] = header_bullshit
                piece_idx += len(temp) 
            else:
                piece_idx += 1
    
    piece_idx = 1
    while piece_idx < len(pieces):
        if not is_header[piece_idx] and not is_header[piece_idx-1]:
            is_header.pop(piece_idx)
            pieces[piece_idx-1] += pieces[piece_idx]
            pieces.pop(piece_idx)
        else:
            piece_idx += 1

    pieces_dict = {'junk':''}
    if is_header[0] == False:
        pieces_dict['junk'] += pieces[0]
        start_idx = 1
    else:
        start_idx = 0
    # govnokod
    if (len(pieces)-start_idx) % 2 != 0:
        pieces.append('')
        
    for piece_idx in range(start_idx,len(pieces),2):
        pieces_dict[pieces[piece_idx]] = pieces[piece_idx+1]

    return pieces_dict

def chopped_to_debug_text(chopped_dict,start='<br/>---',end='---<br/>'):
    result = ''
    for kw in chopped_dict:
        if kw=='junk':
            result+=chopped_dict['junk']
        else:
            result+=start+kw.upper()+end+chopped_dict[kw]
    return result


def kwdict2sectiondict(kw2sectiontext):
    # {keyword : value} - > {section : value}
    result = defaultdict(str)
    for kw in chopped_dict:
        section = kws2sections[kw]
        result[section] += kw2sectiontext[kw] + '\n'
    return result
 
preprocessing_functions = [get_grayscale,         correct_skew,]
apply_preprocessing = lambda input_img: reduce(lambda img, func: func(img), preprocessing_functions, input_img)

def predict(input_image_list, debug=False):
    output_images = [apply_preprocessing(img) for img in input_image_list]
    raw_output = ' '.join([pytesseract.image_to_string(img, lang='rus+eng',) for img in output_images])
    result_dict = chinchoppa(raw_output)
    if debug:
        return prettier_text(chopped_to_debug_text(chopped_dict))
    else:
        return chopped_to_dict(result_dict) 
