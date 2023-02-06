from distutils.log import info
from unittest import result
import cv2 as cv
from cv2 import imshow
import numpy as np
import argparse

from pyparsing import RecursiveGrammarException

path = '/home/leo/Escritorio/Uware/1_imagenes/underwater_id/images/img2.png'
img = cv.imread(path)

# cv.imshow('entrada', img)

# cv.waitKey(0)

# def inpaint(originalImage, mask):
#     [rows, columns, channels] = originalImage.shape
#     result = np.zeros((rows,columns,channels))
#     for row in range(rows):
#         for column in range(columns):
#             if(mask[row,column]==0):
#                 result[row,column] = originalImage[row,column]
#             else:
#                 result[row,column] = (0,0,255)
#     return result
#-------------------------------------  PRE-PROCESAMIENTO   ----------------------

img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
rows, columns, channel  = img_lab.shape
l, a, b = cv.split(img_lab)

l_blur = cv.medianBlur(l, 5)

for row in range(rows):
    for column in range(columns):
        l_in=  img_lab[row, column, 0]
        l_out = (1.5*l_in) - (0.5*l_blur[row, column]) 
        img_lab[row, column, 0] = l_out
        
result = cv.cvtColor(img_lab, cv.COLOR_LAB2BGR)
result_gr = cv.cvtColor(img_lab, cv.COLOR_BGR2GRAY)
# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

img_lab[...,0] = clahe.apply(img_lab[...,0])



result2 = cv.cvtColor(img_lab, cv.COLOR_LAB2BGR)

imshow('resultado preprocesamiento', result)
imshow('entrada', img)
imshow('resultado preprocesamiento CLAHE', result2)





cv.waitKey(0)
# print(np.shape(l))
cv.destroyAllWindows()
#----------------------------------------------------------------------
# img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# #h, s, v = cv.split(img_hsv)

# mask = np.zeros(img_hsv.shape[:2], np.uint8)

# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)

# rect = (190, 5, 470, 470)
# cv.grabCut(img_hsv, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

# mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# img_hsv = img_hsv * mask2[:, :, np.newaxis]

# img_r = cv.inRange(img_hsv, (28, 0, 0), (45, 255, 255))
# img_r = cv.medianBlur(img_r, 5)

# img_rgb = cv.cvtColor(img_r, cv.COLOR_BGR2RGB)
# img_2d = img_rgb.reshape((-1, 3))
# img_2d = np.float32(img_2d)

# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 1.0)
# ret, label, center = cv.kmeans(img_2d, 3, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

# center = np.uint8(center)
# img_segment = center[label.flatten()]
# src = img_segment.reshape((img.shape))

# img_GRAY = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
# _, dst = cv.threshold(img_GRAY, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
# dst = cv.morphologyEx(src, cv.MORPH_CLOSE, element)

# element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
# dst = cv.morphologyEx(dst, cv.MORPH_OPEN, element)

# #element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
# #dst = cv.erode(dst, element)

# #element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
# #dst = cv.dilate(dst, element)

# #--------------------------------------------------------------------------------------------

# cv.imwrite(args.salida, dst)


