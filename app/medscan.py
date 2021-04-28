from PIL import Image
import cv2 
import numpy as np
from scipy.ndimage import interpolation as inter
import pytesseract
import regex as re
from more_itertools import intersperse
from ms_keywords import sections2kws, fields2kws # {str: [str]}
from collections import defaultdict
from functools import reduce
import skimage
from skimage.exposure import equalize_adapthist

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

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
    result = re.sub(r'\s+', cleaner, input_text).strip()
    return result

def transpose_dict(input_dict):
    new_dict = {}
    for group, elements in input_dict.items():
        new_dict.update(dict.fromkeys(elements,group))
    return new_dict

kws2sections = transpose_dict(sections2kws)

def find_fields(text):

    fields = {}
    
    # special case - name; no other fields yet, so it's tolerable
    # later the entire function should be rewritten

    name_kws = fields2kws['name']
    name_kws = sorted(name_kws, key=len, reverse=True)
    for kw in name_kws:
        # first part: looks for the keyword
        # second part: any amount of whitespace characters on either sides of three words
        # the three words are either n uppercase characters (n>=1) or 1 uppercase character and n lowercase characters (n>=0)
        exp = f'{kw}[\n:]*' + r'\s*((\p{Lu}+\s*){3}|(\p{Lu}\p{Ll}*\s*){3})'
        name_match = re.search(exp, text)
        if name_match:
            name = name_match.groups('')[0]
            name = prettier_text(name)
            name = ''.join([c if c!='\n' else ' ' for c in name])
            fields['name'] = name
            break
    return fields

def process_discharge(text):
    sections_kws = kws2sections.keys()
    sections_kws = sorted(sections_kws, key=len, reverse=True)
    # break down text into keyword and surrounding text until we run out of keywords
    # store text in original order in "pieces" list, and its type (header/keyword) in "is_kw"
    pieces = [text]
    is_kw = [False]
    for kw in sections_kws:
        piece_idx = 0
        while piece_idx < len(pieces):
            if not is_kw[piece_idx]:
                # TODO: test if it still works without the next line
                # kw_ = kw[0].upper() + kw[1:]
                # keyword is at least: prefixed with newline, postfixed with newline, or postfixed with colon
                sub_pieces = re.split(f'\n{kw}|{kw}[\n:]', pieces[piece_idx]) 
                # remove non-alphanumeric symbols in the start of the string
                # TODO: test if it is useful
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
    # now it's always [..., header, text, header, ...]

    # make sure pieces is a list of consecutive pairs: [header, text, ..., header, text]
    if is_kw[0] == False:
        pieces.insert(0, 'Junk') # Sic (because of for loop in ms_keywords)
        is_kw.insert(0, True)
    if len(pieces) % 2 != 0:
        pieces.append('')

    # finally put them in our dict 
    kw2sectiontext = {}   
    for piece_idx in range(0,len(pieces),2):
        kw2sectiontext[pieces[piece_idx]] = pieces[piece_idx+1]

    # transition from keywords to internal section names
    result = kwdict2sectiondict(kw2sectiontext)
    for section in result:
        result[section] = prettier_text(result[section]) # prettify text

    fields = find_fields(text)
    result.update(fields) # add fields to the result (such as name and date)

    return result


def kwdict2sectiondict(kw2sectiontext):
    # {keyword : value} - > {section : value}
    sectiondict = defaultdict(str)
    for kw in kw2sectiontext:
        section = kws2sections[kw]
        sectiondict[section] += kw2sectiontext[kw] + '\n'
    return sectiondict

# checksum? more than one? raise error TODO also in insurance iterate over matches to see if one of them checks out
# make sure that symbols on either side of the numbers are not numbers (add to the regexp) TODO
noise = '''."'`,\u00B0'''
def process_insurance(text):
    text = re.sub(f'[ \t{noise}]', '', text) # maybe remove some of these later TODO
    match = re.search(r'(\D|^)\d{16}(\D|$)', text)
    if match:
        return {'insurance': match.group()}
    return {'error':'not found', 'text' : text}

def process_snils(text):
    text = re.sub(f'[ \t{noise}]', '', text)
    match = re.search(r'\d{3}-\d{3}-(\d{5}|\d{3}-\d{2})', text)
    if match:
        return {'snils': match.group().replace('-','')}
    return {'error':'not found', 'text' : text}

preprocessing_functions = [get_grayscale, correct_skew,]
apply_preprocessing = lambda input_img: reduce(lambda img, func: func(img), preprocessing_functions, input_img)

def preprocess_image(img):
    img = get_grayscale(img)
    img = correct_skew(img)
    img = skimage.util.img_as_ubyte(equalize_adapthist(img, nbins = 2))
    cv2.imwrite("files/lastupload.png", img) 
    return img

def text_recognition(input_images, doc_type):
    preprocessed_images = [preprocess_image(img) for img in input_images]
    text = ' '.join([pytesseract.image_to_string(img, lang='rus+eng', config='--oem 1') for img in preprocessed_images])

    if doc_type == 'discharge':
        return process_discharge(text)
    elif doc_type == 'insurance':
        return process_insurance(text)
    elif doc_type == 'snils':
        return process_snils(text)
