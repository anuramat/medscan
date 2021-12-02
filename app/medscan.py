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
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion, binary_opening, binary_closing, binary_dilation
from skimage.filters import threshold_otsu, unsharp_mask, median, gaussian
from skimage.io import imsave
from skimage.restoration import denoise_bilateral, denoise_wavelet
from skimage.morphology import dilation, erosion, opening, closing
from skimage.morphology import disk

from skimage.util import compare_images

# TODO move
debug = True
debug_folder = 'files/'

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

def wrapped_ocr(img, lang=None, whitelist=None):
    if not lang:
        lang = 'rus+eng'
    config = '--oem 1 --psm 1'
    if whitelist:
        config += ' -c tessedit_char_whitelist=' + whitelist
    text = pytesseract.image_to_string(img, lang=lang, config=config)
    return text

def process_discharge(input_images):
    preprocessed_images = [preprocess_image(img, 256) for img in input_images]
    text = ' '.join([wrapped_ocr(img) for img in preprocessed_images])

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
def process_insurance(img):
    img = get_grayscale(img)
    img = correct_skew(img)
    
    text = wrapped_ocr(img, lang='rus', whitelist='1234567890')
    match = re.search(r'(\D|^)\d{16}(\D|$)', text)
    if debug:
        imsave(debug_folder + '/lastupload.png', img)
        add_fail(text) 
    if match:
        return {'insurance': match.group().strip()}
    return {'error':'not found', 'text' : text}

def process_snils(img):
    # fine font
    img = get_grayscale(img)
    img = correct_skew(img)
    initial_img = img.copy() 
    img = skimage.util.img_as_ubyte(equalize_adapthist(img, kernel_size=None,nbins=16))
    img = skimage.morphology.opening(img, skimage.morphology.disk(3))
    img = skimage.util.img_as_ubyte(unsharp_mask(img, 10, 3))
    img = skimage.morphology.erosion(img, skimage.morphology.disk(3))
    if debug:
        imsave(debug_folder + '/lastupload.png', img)
    text = wrapped_ocr(img, lang='rus', whitelist='1234567890-')
    match = re.search(r'\d{3}-+\d{3}-+(\d{5}|\d{3}-+\d{2})', text)
    if match:
        return {'snils': re.sub(r'\D', '', match.group())}

    
    # thicc font 
    img = initial_img
    img = skimage.util.img_as_ubyte(equalize_adapthist(img, kernel_size=None,nbins=16))
    img = skimage.morphology.opening(img, skimage.morphology.disk(2))
    img = skimage.util.img_as_ubyte(unsharp_mask(img, 5, 2))
    if debug:
        imsave(debug_folder + '/lastupload2.png', img)
    text = wrapped_ocr(img, lang='rus', whitelist='1234567890-')
    match = re.search(r'\d{3}-+\d{3}-+(\d{5}|\d{3}-+\d{2})', text)
    if match:
        return {'snils': re.sub(r'\D', '', match.group())}


    # fail
    if debug:
        add_fail(text) 
    return {'error':'not found', 'text' : text}

def mrz_checksum(value):
    weights = [7,3,1]
    total = 0
    for i in range(len(value)):
        if value[i] == '<':
            continue
        if not value[i].isdigit():
            return 'error' 
        total += int(value[i]) * weights[i%3]
    return str(total%10)
eng_alphabet = 'ABVGDE2JZIQKLMNOPRSTUFHC34WXY9678'
rus_alphabet = 'АБВГДЕЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
mrz_dict = {eng_alphabet[i]: rus_alphabet[i] for i in range(len(eng_alphabet))}
def process_passport(img):
    img = get_grayscale(img)
    img = correct_skew(img)
    input_img = img.copy() 
    height = 600
    ratio = img.shape[1]/img.shape[0] # width/height
    width = int(ratio*height)
    dim = (width, height)
    # resize image, because we have fixed size kernels
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, rectKernel)
    img = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    img = np.absolute(img)
    (minVal, maxVal) = (np.min(img), np.max(img))
    img = (255 * ((img - minVal) / (maxVal - minVal))).astype("uint8")

    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, rectKernel)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, sqKernel)
    img = cv2.erode(img, None, iterations=4)

    cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    mrz_img = None
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        crWidth = w / width
        # check to see if the aspect ratio and coverage width are within
        # acceptable criteria
        if ar > 5 and crWidth > 0.75:
            # pad the bounding box since we applied erosions and now need
            # to re-grow it
            pX = int((x + w) * 0.03)
            pY = int((y + h) * 0.03)
            (x, y) = (x - pX, y - pY)
            (w, h) = (w + (pX * 2), h + (pY * 2))
            # extract the ROI from the image
            factor = input_img.shape[0]/height
            y = int(y*factor)
            x = int(x*factor)
            h = int(h*factor)
            w = int(w*factor)
            mrz_img = input_img[y:y + h, x:x + w].copy()
            break
    if type(mrz_img) is not np.ndarray or np.prod(mrz_img.shape) == 0:
        return {'error': 'MRZ was not found'}
    img = mrz_img.copy()

    if debug:
        cv2.imwrite(debug_folder + "/lastupload.png", input_img) 
    whitelist = '<ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
    # OCR STARTS HERE
    input_img = img.copy()
    img = dilation(img)
    img = median(img, np.ones((10,10)))
    img = 1-compare_images(img,input_img, method='diff')

    img = skimage.util.img_as_ubyte(img)
    # OCR END
    
    if debug:
        cv2.imwrite(debug_folder + "/lastupload2.png", img) 
    mrz = wrapped_ocr(img, lang='eng', whitelist=whitelist)
    
    mrz = mrz.strip() 
    mrz_lines = mrz.split('\n')
     
    if len(mrz_lines) != 2 or len(mrz) < 60:
        return {'error': 'xd', 'text': mrz}
    
    for i in range(2):
        mrz_lines[i] = mrz_lines[i].ljust(44, '<')[:44]
        
    mrz = ''.join(mrz_lines)




    # MRZ parser
    # TODO: make robust in terms of scanos 
    # TODO: add check: if len(mrz)==88(+1)
    result = {}
    if debug:
        result['mrz_line'] = mrz_lines
    name = ''
    for i in ' '.join(mrz[5:44].split('<')):
        if i in eng_alphabet:
            name += mrz_dict[i]
        else:
            name += ' '
    name = re.sub('[ ]+',' ',name).rstrip()
    result['name'] = name
    
    mrz = mrz[44:]
    reverse_date = lambda x: x[4:] + x[2:4] + x[:2]
    
    result['id_series'] = mrz[:3] + mrz[28]
    result['id_no'] = mrz[3:9]
    result['corr_1'] = mrz_checksum(mrz[:9]) == mrz[9]
    result['dob'] = reverse_date(mrz[13:19])
    result['corr_2'] = mrz_checksum(mrz[13:19]) == mrz[19]
    result['sex'] = mrz[20]
    result['issued'] = reverse_date(mrz[29:35])
    result['authority_code'] = mrz[35:41]
    result['corr_3'] = mrz_checksum(mrz[28:41]) == mrz[42]
    result['corr_final'] = mrz_checksum(mrz[:10]+mrz[13:20]+mrz[21:28]+mrz[28:43]) == mrz[43]
    
    
    return result


    

def text_recognition(input_images, doc_type):
    if doc_type == 'passport':
        return process_passport(input_images[0])
    elif doc_type == 'discharge':
        return process_discharge(input_images)
    elif doc_type == 'insurance':
        return process_insurance(input_images[0])
    elif doc_type == 'snils':
        return process_snils(input_images[0])

def add_fail(text):
    # parameterize the folder TODO
    res = '\n\n\n****upload start****\n\n\n' + text + '\n\n\n****upload end\n\n\n'
    with open(debug_folder + '/fails.txt', 'a+') as textfile:
        textfile.write(res) 
