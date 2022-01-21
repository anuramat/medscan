# TODO remove unnecessary dependencies
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
from skimage.morphology import dilation, erosion, opening, closing, black_tophat
from skimage.morphology import disk
from skimage.transform import rescale, resize
from skimage.util import compare_images
from skimage.measure import find_contours

import easyocr
reader = easyocr.Reader(['en'])

# TODO move to dotfile or smth
debug = True
debug_folder = 'files/'
def dsave(img, name):
    if debug:
        imsave(debug_folder + f'/{name}.png', img)
    

# TODO move to utils or smth
def wrapped_ocr(img, lang=None, whitelist=None):
    if not lang:
        lang = 'rus+eng'
    config = '--oem 1 --psm 1'
    if whitelist:
        config += ' -c tessedit_char_whitelist=' + whitelist
    text = pytesseract.image_to_string(img, lang=lang, config=config)
    return text

# TODO use skimage
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# TODO use skimage
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# TODO use skimage, rewrite
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

# TODO refactor from here

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


def process_discharge(input_images):
    
    text_pages = [] 
    for img in input_images:
        img = get_grayscale(img)
        img = correct_skew(img)
        # img = skimage.util.img_as_ubyte(equalize_adapthist(img, kernel_size=None,nbins=256))
        text_pages.append(wrapped_ocr(img))
    text = '\n'.join(text_pages)

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
                '.(?=\n{kw}|{kw}[\n:])'
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

# TODO refactor to here

def process_insurance(img):
    img = get_grayscale(img)

    # try scaling to 1k by whatever if accuracy drops
    img = np.asarray(img)
    img = correct_skew(img)

    input_img = img.copy() # save for when we find the numbers

    # find the barcode under the numbers
    # first use blur and morphological operations to turn the barcode into a solid rectangle
    img = black_tophat(img, np.ones((1,200)))
    img = median(img, np.ones((1,20)))
    img = closing(img, np.ones((20,20)))

    img = threshold_otsu(img)<img

    img = closing(img, np.ones((20,1)))  # vertical fill
    img = opening(img, np.ones((20,20))) # shrink stuff

    # find the edges of the resulting rectangle
    contours = find_contours(img)

    '''
    # save visualization of contours just in case
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    for contour in contours:
        ax.plot(contour[:,1], contour[:, 0], linewidth=2)
    plt.show()
    '''

    # now we calculate the bounding boxes
    bounding_boxes = [] # here we store all the bounding boxes, that correspond to the found contours
    matches = [] # here we store just the candidates to be a barcode (we hope that we find just one)

    for contour in contours:
        
        # find a box and calculate its properties
        box = {'x_min': np.min(contour[:,1]),
                'x_max': np.max(contour[:,1]),
                'y_min': np.min(contour[:,0]),
                'y_max': np.max(contour[:,0])}
        width = box['width'] = box['x_max'] - box['x_min']
        height = box['height'] = box['y_max'] - box['y_min']
        area = box['area'] = width*height
        ar = box['ar'] = width/height
        ira = box['ira'] = np.product(img.shape)/area
        
        # save it to the list
        bounding_boxes.append(box)
    # TODO figure how the fuck this works    
        # check if it matches the criteria
        ira_match = 40<ira<55 or True # TODO remove HACK 
        ar_match = 4<ar<5
        if ira_match and ar_match:
            matches.append(box)
    
    # check if theres a clear winner
    #if len(matches) != 1:
    #    return {'error': 'more than one candidate for barcode'}
    matches = sorted(matches, key = lambda x: x['ar']) 

    # now lets find the part of the image with numbers by just flipping our barcode rectangle once
    numbers_box = matches[0]
    numbers_box['y_min'] -= numbers_box['height']
    numbers_box['y_max'] -= numbers_box['height']
    for key in ['x_min','x_max','y_min','y_max']:
        numbers_box[key] = int(numbers_box[key])
    numbers_img = input_img[numbers_box['y_min']: numbers_box['y_max'], numbers_box['x_min']:numbers_box['x_max']]
    
    # start OCR
    text = wrapped_ocr(numbers_img, lang='eng', whitelist='1234567890')
    text = re.sub(r'\D', '', text)
    if len(text) == 16:
        return {'insurance': text}
    return {'error':'error', 'text' : text}

def process_snils(img):
    snils_regex = re.compile(r'(\D|^)\d{3}[- ]*\d{3}[- ]*\d{3}[- ]*\d{2}(\D|$)')
    img = get_grayscale(img)
    img = correct_skew(img)
    initial_img = img.copy() 


    # fine font
    img = skimage.util.img_as_ubyte(equalize_adapthist(img, kernel_size=None,nbins=16))
    img = skimage.morphology.opening(img, skimage.morphology.disk(3))
    img = skimage.util.img_as_ubyte(unsharp_mask(img, 10, 3))
    img = skimage.morphology.erosion(img, skimage.morphology.disk(3))
    text = wrapped_ocr(img, lang='rus', whitelist='1234567890-')
    match = re.search(snils_regex, text)
    if match:
        return {'type': 'fine', 'snils': re.sub(r'\D', '', match.group())}

     
    # thicc font 
    # TODO try without rescaling
    initial_img = rescale(initial_img, 0.7, anti_aliasing=True) # TODO fix later? find a magic value (0.7 as of now)
    initial_height = initial_img.shape[0]
    target_height = 800
    scaling_factor = target_height/initial_height
    initial_img = rescale(initial_img, scaling_factor, anti_aliasing=True)
    img = initial_img.copy()
    img = skimage.util.img_as_ubyte(equalize_adapthist(img, kernel_size=None,nbins=16))
    img = dilation(img, disk(3))
    img = gaussian(img, 3)
    img = median(img, np.ones((10,10)))
    img = 1-compare_images(img,initial_img, method='diff')
    img = skimage.util.img_as_ubyte(unsharp_mask(img, 5, 2))  
    #img = rescale(img, 2, anti_aliasing=True) # TODO readd in case something breaks?
    text = wrapped_ocr(img, lang='rus', whitelist='1234567890-')
    match = re.search(snils_regex, text)
    if match:
        return {'type':'thicc', 'snils': re.sub(r'\D', '', match.group())}


    # fail
    return {'error':'error', 'text' : text}

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
    # img = skimage.util.img_as_ubyte(equalize_adapthist(img, kernel_size=None,nbins=256))

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
    img = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
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
        #return {'error': 'MRZ was not found'}
        # brute force fallback: get bottom
        approx_mrz_start = int(0.8*input_img.shape[0])
        mrz_img = input_img[approx_mrz_start:, :]
    img = mrz_img.copy()

    whitelist = '<ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
    # OCR STARTS HERE
    target_height = 200
    initial_height = img.shape[0]
    scaling_factor = target_height/height
    #img = rescale(img, scaling_factor) # TODO turn on and check

    input_img = img.copy()
    img = dilation(img)
    img = median(img, np.ones((10,10)))
    img = 1-compare_images(img,input_img, method='diff')

    img = skimage.util.img_as_ubyte(img)
    # OCR END
    
    mrz = wrapped_ocr(img, lang='eng', whitelist=whitelist)
    
    mrz = mrz.strip()
    mrz_lines = mrz.split() # TODO parse propersly: ignore spaces in the middle of the lines
     
    if len(mrz_lines) != 2 or len(mrz) < 60:
        return {'error': 'error', 'text': mrz}
    
    for i in range(2):
        mrz_lines[i] = mrz_lines[i].ljust(44, '<')[:44]
        
    mrz = ''.join(mrz_lines)




    # MRZ parser
    # TODO: make robust in terms of scanos 
    # TODO: add check: if len(mrz)==88(+1)
    eng_alphabet = 'ABVGDE2JZIQKLMNOPRSTUFHC34WXY9678'
    rus_alphabet = 'АБВГДЕЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    mrz_dict = {eng_alphabet[i]: rus_alphabet[i] for i in range(len(eng_alphabet))}

    result = {}
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
    add_dots = lambda x: x[:2] + '.' + x[2:4] + '.' + x[4:]
    format_datestr = lambda x: add_dots(reverse_date(x))

    result['id_series'] = mrz[:3] + mrz[28]
    result['id_no'] = mrz[3:9]
    result['corr_1'] = mrz_checksum(mrz[:9]) == mrz[9]
    result['dob'] = format_datestr(mrz[13:19])
    result['corr_2'] = mrz_checksum(mrz[13:19]) == mrz[19]
    result['sex'] = mrz[20]
    result['issued'] = format_datestr(mrz[29:35])
    result['authority_code'] = mrz[35:41]
    result['corr_3'] = mrz_checksum(mrz[28:41]) == mrz[42]
    result['corr_final'] = mrz_checksum(mrz[:10]+mrz[13:20]+mrz[21:28]+mrz[28:43]) == mrz[43]
    
    
    return result


    

def text_recognition(input_images, doc_type):
    if doc_type == 'passport':
        return process_passport(input_images[0])
    if doc_type == 'discharge':
        return process_discharge(input_images)
    if doc_type == 'insurance':
        return process_insurance(input_images[0])
    if doc_type == 'snils':
        return process_snils(input_images[0])
    if doc_type == 'card':
        return process_card(input_images[0])
    if doc_type == 'emb':
        return process_emb(input_images[0])


def process_card(img):
    img = get_grayscale(img)
    img = correct_skew(img)
    input_img = img.copy()
    img = dilation(img)
    img = median(img, np.ones((20,20)))
    img = 1-compare_images(img,input_img, method='diff')
    img = skimage.util.img_as_ubyte(img)
    dsave(img, 'card')

    text = wrapped_ocr(img, whitelist = '1234567890')
    lines = text.split()
    result = {}
    for line in lines[::-1]:
        numbers = re.sub('\D', '', line)
        if len(numbers) == 16:
            result['insurance'] = numbers
            break
    if 'insurance' not in result:
        numbers = re.sub('\D', '', text)
        if len(numbers) == 16:
            result['insurance'] = numbers
        
    if 'insurance' not in result:
        result['error'] = 'error'        
        result['text'] = text 
    return result
    

def process_emb(img):
    img = get_grayscale(img)
    img = correct_skew(img)

    img = skimage.util.img_as_ubyte(img)

    easyocr_result = reader.readtext(img, allowlist = '0123456789', width_ths = 1)
    easyocr_result = sorted(easyocr_result, key = lambda x: x[1], reverse=True)
    result = {}
    for block in easyocr_result:
        if len(block[1]) == 16:
            result['insurance'] = block[1]
            result['confidence'] = block[2]

    if 'insurance' not in result:
        result['error'] = 'error' 
        result['text'] = [block[1] for block in easyocr_result]
    return result
