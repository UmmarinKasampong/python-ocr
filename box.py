import cv2
import pytesseract
import numpy as np

def detectColor(img , hsv):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # ปรับภาพให้เป็นสีสว่าง
    # cv2.imshow("hsv" , imgHSV)
    lower = np.array([hsv[0] , hsv[2] , hsv[4]])
    upper = np.array([hsv[1] , hsv[3] , hsv[5]])
    mask = cv2.inRange(imgHSV , lower , upper)
    # cv2.imshow("mask" , mask)
    # *****
    imgResult = cv2.bitwise_and(img , img , mask=mask)
    cv2.imshow("imgResult" , imgResult)
    return imgResult



# def threshold_image(img_src):
#     """Grayscale image and apply Otsu's threshold"""
#     # Grayscale
#     img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
#     # Binarisation and Otsu's threshold
#     _, img_thresh = cv2.threshold(
#         img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     return img_thresh, img_gray

# def getContours(img , imgDraw , cThr=[100,100] , showCanny=False , minArea=1000 , filter=0 , draw=False):
    imgDraw = imgDraw.copy()
    imgGray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray , (5,5) , 1)
    # print(cThr[1])
    imgCanny = cv2.Canny(imgBlur , cThr[0] , cThr[1])
    kernel= np.array((10,10))
    imgDial = cv2.dilate(imgCanny , kernel , iterations=1)
    imgClose = cv2.morphologyEx(imgDial , cv2.MORPH_CLOSE , kernel)
    # cv2.imshow("Gray" , imgGray)
    

    if showCanny: cv2.imshow("Canny" , imgClose)
    contours , hiearchy = cv2.findContours(imgClose , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )
    finalCountours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i , True)
            approx = cv2.approxPolyDP(i , 0.02 * peri , True)
            bbox = cv2.boundingRect(approx)
            if filter > 0 :
                if len(approx) == filter:
                    finalCountours.append([len(approx) , area , approx , bbox , i])
            else :
                finalCountours.append([len(approx) , area , approx , bbox , i])
    finalCountours = sorted(finalCountours , key=lambda x : x[1] , reverse=True)
    if draw:
        for con in finalCountours:
            x , y , w , h = con[3]
            cv2.rectangle(imgDraw , (x , y) , (x + w , y + h) , (255 , 0 , 255) , 3)
    # print(contours)
    return imgDraw , finalCountours


