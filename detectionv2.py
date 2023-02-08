from distutils.log import info
from unittest import result
import cv2 as cv
from cv2 import imshow
from cv2 import split
from cv2 import Sobel
import numpy as np
import argparse

from pyparsing import RecursiveGrammarException

# path = '/home/leo/Escritorio/Uware/1_imagenes/underwater_id/images/img1.webp'
path = '/home/leo/Escritorio/Uware/1_imagenes/underwater_id/images/img2.png'
img = cv.imread(path)


#-------------------------------------  PRE-PROCESAMIENTO   ----------------------

img_lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
rows, columns, channel  = img_lab.shape
l, a, b = cv.split(img_lab)

l_blur = cv.medianBlur(l, 5)

for row in range(rows):
    for column in range(columns):
        l_in=  img_lab[row, column, 0]
        l_out = (1.5*l_in) - (0.5*l_blur[row, column]) 
        img_lab[row, column, 0] = l_out
        
result = cv.cvtColor(img_lab, cv.COLOR_LAB2RGB)
result_gr = cv.cvtColor(img_lab, cv.COLOR_RGB2GRAY)
# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

img_lab[...,0] = clahe.apply(img_lab[...,0])



result_prep = cv.cvtColor(img_lab, cv.COLOR_LAB2RGB)

#--------------Clusterización -----------------------------------------#


rows, columns, channel  = result_prep.shape
# features = np.zeros(shape=(rows,columns, 4))


 # Eliminacion del ruido de fondo
img_rgb = cv.bilateralFilter(result_prep, 15, 90, 90)
img_hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)

ddepth = cv.CV_16S

#Reduccion del canal H en el espacio HSV
h, s, v = cv.split(img_hsv)
h = h - 20
img_hsv_2 = cv.merge((h,s,v))

img_rgb = cv.cvtColor(img_hsv_2, cv.COLOR_HSV2RGB)
# img_gray = cv.cvtColor(img_bgr, cv.COLOR_RGB2GRAY)


img_meanshift = cv.pyrMeanShiftFiltering(img_rgb, 20, 20)

########################################################################

# Forma del filtro
erosion_type = cv.MORPH_RECT
erosion_type2 = cv.MORPH_ELLIPSE
# El último parámetro es el tamaño del filtro, en este caso 5x5
element = cv.getStructuringElement(erosion_type, (6,6)) 
element2 = cv.getStructuringElement(erosion_type2, (4,4)) 


# dst = cv.erode(dst,element2)
dst = cv.morphologyEx(img_meanshift, cv.MORPH_CLOSE, element2)
dst2 = cv.morphologyEx(dst, cv.MORPH_CLOSE, element2)
dst3 = cv.dilate(dst2, element)


   
# cv2.cvtColor is applied over the
# image input with applied parameters
# to convert the image in grayscale 
# img = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
   
# # applying different thresholding 
# # techniques on the input image
# thresh1 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
#                                           cv.THRESH_BINARY, 199, 1)
  
# thresh2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                           cv.THRESH_BINARY, 199, 1)
  
# # the window showing output images
# # with the corresponding thresholding 
# # techniques applied to the input image
# cv.imshow('Adaptive Mean', thresh1)
# cv.imshow('Adaptive Gaussian', thresh2)





# img_hsv = cv.cvtColor(dst, cv.COLOR_RGB2HSV)
# Z = img_hsv.reshape((-1,3))
# # convert to np.float32
# Z = np.float32(Z)
# # define criteria, number of clusters(K) and apply kmeans()
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# K = 5
# ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
# # Now convert back into uint8, and make original image
# center = np.uint8(center)
# res = center[label.flatten()]
# res3 = res.reshape((img_hsv.shape))


# imshow('resultado preprocesamiento', result)
# imshow('entrada', result2)
# imshow('hsv1', img_hsv)
# imshow('hsv2', img_hsv_2)
# imshow('rgb', img_rgb)
imshow('menashift', img_meanshift)
imshow('dst', dst)
imshow('dst2', dst2)
imshow('dst3', dst3)
# imshow('hsv2', res3)

cv.waitKey(0)
# # print(np.shape(l))
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


