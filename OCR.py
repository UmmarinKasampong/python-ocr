import cv2
import pytesseract
from func import *

path = 'input/1.jpeg'



img = cv2.imread(path)
# resize = cv2.resize(img, (800 , 900))

# แปลงภาพ เป็นขาวดำ binary 0 , 1
img_thresh, img_gray = threshold_image(img)


   # yellow highlight colour range
hsv_lower = [22, 30, 30]
hsv_upper = [45, 255, 255]

    # Color segmentation

# สร้าง mask แยกส่วนที่ hight light กับภาพจริง
img_mask, img_hsv = mask_image(img, hsv_lower, hsv_upper)

    # Noise reduction
img_mask_denoised = denoise_image(img_mask)

    # get ocr data
data_ocr = image_to_data(img_thresh)


data_ocr = find_highlighted_words(img_mask_denoised, data_ocr, threshold_percentage=25)
str_highlight = words_to_string(data_ocr)

output_text_file = 'output/highlighted_text.txt'
export_text_to_file(str_highlight, output_text_file)

# print(data_ocr)
# cv2.imshow("img_mask" , img_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()



