import cv2
import pytesseract
import numpy as np
from pytesseract import Output
import re

class Levels:
    PAGE = 1
    BLOCK = 2
    PARAGRAPH = 3
    LINE = 4
    WORD = 5


def export_text_to_file(text, file_path):
    with open(file_path, 'w') as file:
        file.write(text)

        

def threshold_image(img_src):
    """Grayscale image and apply Otsu's threshold"""
    # Grayscale
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    # Binarisation and Otsu's threshold
    _, img_thresh = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img_thresh, img_gray


def image_to_data(img_src):
    return pytesseract.image_to_data(
        img_src, lang='tha+eng', config='--psm 6', output_type=Output.DICT)


def mask_image(img_src, lower, upper):
    """Convert image from RGB to HSV and create a mask for given lower and upper boundaries."""
    # RGB to HSV color space conversion
    img_hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)
    hsv_lower = np.array(lower, np.uint8)  # Lower HSV value
    hsv_upper = np.array(upper, np.uint8)  # Upper HSV value

    # Color segmentation with lower and upper threshold ranges to obtain a binary image
    img_mask = cv2.inRange(img_hsv, hsv_lower, hsv_upper)

    return img_mask, img_hsv


def denoise_image(img_src):
    """Denoise image with a morphological transformation."""

    # Morphological transformations to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    img_denoise = cv2.morphologyEx(
        img_src, cv2.MORPH_OPEN, kernel, iterations=1)


    return img_denoise #, contours, hierarchy, img_contour



def find_highlighted_words(img_mask, data_ocr, threshold_percentage=25):
    """Find highlighted words by calculating how much of the words area contains white pixels compared to balack pixels."""

    # Initiliaze new column for highlight indicator
    data_ocr['highlighted'] = [False] * len(data_ocr['text'])

    for i in range(len(data_ocr['text'])):
        # Get bounding box position and size of word
        (x, y, w, h) = (data_ocr['left'][i], data_ocr['top']
                        [i], data_ocr['width'][i], data_ocr['height'][i])
        # Calculate threshold number of pixels for the area of the bounding box
        rect_threshold = (w * h * threshold_percentage) / 100
        # Select region of interest from image mask
        img_roi = img_mask[y:y+h, x:x+w]
        # Count white pixels in ROI
        count = cv2.countNonZero(img_roi)
        # Set word as highlighted if its white pixels exceeds the threshold value
        if count > rect_threshold:
            data_ocr['highlighted'][i] = True

    return data_ocr



def words_to_string(data_ocr):
    word_list = []
    line_breaks = (Levels.PAGE, Levels.BLOCK, Levels.PARAGRAPH, Levels.LINE)

    for i in range(len(data_ocr['text'])):
        # print("Level: {}; Page: {}; Block: {}; Paragraph: {}; Line: {}; Word: {}; Highlighted: {} Text: {}".format(
        #     data_ocr['level'][i],
        #     data_ocr['page_num'][i],
        #     data_ocr['block_num'][i],
        #     data_ocr['par_num'][i],
        #     data_ocr['line_num'][i],
        #     data_ocr['word_num'][i],
        #     data_ocr['highlighted'][i],
        #     data_ocr['text'][i]))

        if data_ocr['level'][i] in line_breaks:
            word_list.append("\n")
            continue

        text = data_ocr['text'][i].strip()

        if text and data_ocr['highlighted'][i]:
            word_list.append(text + " ")

    # concat all words into one string
    word_string = "".join(word_list)
    # repalce multiple consecutive newlines with one single newline
    word_string = re.sub(r'\n+', '\n', word_string).strip()

    return word_string