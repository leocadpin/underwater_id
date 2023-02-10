from distutils.log import info
from unittest import result
import cv2 as cv
from cv2 import imshow
from cv2 import split
from cv2 import Sobel
from cv2 import merge
import numpy as np
import argparse
import math
import os
from pyparsing import RecursiveGrammarException
from torch import sqrt_

# path = '/home/leo/Escritorio/Uware/1_imagenes/underwater_id/images/img1.webp'
# path = '/home/leo/Escritorio/Uware/1_imagenes/underwater_id/images/img2.png'
# path = '/home/leo/Escritorio/Uware/1_imagenes/underwater_id/images/img3.png'
# path = '/home/leo/Escritorio/Uware/1_imagenes/underwater_id/images/img4.png'
# path = '/home/leo/Escritorio/Uware/1_imagenes/underwater_id/images/img5.png'
# path = '/home/leo/Escritorio/Uware/1_imagenes/underwater_id/images/img6.png'
# path = '/home/leo/Escritorio/Uware/1_imagenes/underwater_id/images/img7.png'

# Read the image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "nombre de la imagen")
args = vars(ap.parse_args())
path = os.path.join('images', args['image'])
img = cv.imread(path)

# print('Original Dimensions : ',img.shape)
 
# scale_percent = 40 # percent of original size
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)
  
# # resize image
# resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
 
# print('Resized Dimensions : ',resized.shape)
 
# cv.imshow("Resized image", resized)

# img= resized

def morphology_filters(edges):
        # Forma del filtro
    erosion_type = cv.MORPH_RECT
    erosion_type2 = cv.MORPH_ELLIPSE
    # El último parámetro es el tamaño del filtro, en este caso 5x5
    # element = cv.getStructuringElement(erosion_type, (4,4)) 
    # element2 = cv.getStructuringElement(erosion_type2, (4,4)) 
    element3 = cv.getStructuringElement(erosion_type2, (1,1))
    element4 = cv.getStructuringElement(erosion_type, (1,1)) 
    # dst = cv.erode(dst,element2)
    # dst = cv.morphologyEx(img_meanshift, cv.MORPH_CLOSE, element2)
    # dst2 = cv.morphologyEx(dst, cv.MORPH_CLOSE, element2)
    edges_erode = cv.dilate(edges, element4)
    edges_erode = cv.dilate(edges_erode, element4)
  
    edges_erode = cv.erode(edges_erode, element3)
   
    return edges_erode
#-------------------------------------  PRE-PROCESAMIENTO   ----------------------
def preprocessing(img):
    # img2 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
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
    return result_prep

result_prep = preprocessing(img)

#--------------Clusterización -----------------------------------------#

#Primero se crea un vector de caracteristicas
rows, columns, channel  = result_prep.shape
features = np.zeros(shape=(rows,columns, 6))

img_rgb = cv.bilateralFilter(result_prep, 15, 90, 90)
img_rgb = cv.pyrMeanShiftFiltering(img_rgb, 20, 20)
# img_rgb = result_prep
img_hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)


def splitted_sobels(img_rgb):
    # Conversion a escala de grises 
    r_img , g_img, b_img = split(img_rgb)


    # imshow('r_channel', r_img)
    # imshow('g_channel', g_img)
    # imshow('b_channel', b_img)

    # r_img = img_rgb.copy()
    # r_img[:,:,0] =0

    # g_img = img_rgb.copy()
    # # g_img[:,:, 1] =0
    # g_img[:,:, 1] =0

    # b_img = img_rgb.copy()
    # b_img[:,:,2] =0


    # imshow('r_img_channel', r_img)
    # imshow('g_img_channel', g_img)
    # imshow('b_img_channel', b_img)


    # #Gradientes
    ddepth = cv.CV_16S

    r_grad_x = cv.Sobel(r_img, ddepth, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    r_grad_y = cv.Sobel(r_img, ddepth, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        
    r_abs_grad_x = cv.convertScaleAbs(r_grad_x)
    r_abs_grad_y = cv.convertScaleAbs(r_grad_y)

    g_grad_x = cv.Sobel(g_img, ddepth, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    g_grad_y = cv.Sobel(g_img, ddepth, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        
    g_abs_grad_x = cv.convertScaleAbs(g_grad_x)
    g_abs_grad_y = cv.convertScaleAbs(g_grad_y)

    b_grad_x = cv.Sobel(b_img, ddepth, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    b_grad_y = cv.Sobel(b_img, ddepth, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        
    b_abs_grad_x = cv.convertScaleAbs(b_grad_x)
    b_abs_grad_y = cv.convertScaleAbs(b_grad_y)  


    #TODO: IMPLEMENTAR CALCULO DE SOBEL CON RAIZ CUADRADA   
    # sqr_grad = ((grad_x ** 2) + (grad_y ** 2))**0,5

    r_abs_grad = cv.addWeighted(r_abs_grad_x, 0.5, r_abs_grad_y, 0.5, 0)
    g_abs_grad = cv.addWeighted(g_abs_grad_x, 0.5, g_abs_grad_y, 0.5, 0)
    b_abs_grad = cv.addWeighted(b_abs_grad_x, 0.5, b_abs_grad_y, 0.5, 0)

    # imshow('r_img_channel', r_abs_grad)
    # imshow('g_img_channel', g_abs_grad)
    # imshow('b_img_channel', b_abs_grad)
    # # print(np.shape(features))
    return r_abs_grad, g_abs_grad, b_abs_grad


r_abs_grad, g_abs_grad, b_abs_grad = splitted_sobels(img_rgb)

def multi_edges(r_abs_grad, g_abs_grad, b_abs_grad):
    aperture_size = 3
    histeresys_min_thres = 100
    histeresys_max_thres = 180
    r_edges = cv.Canny(r_abs_grad, histeresys_min_thres, histeresys_max_thres, None, aperture_size)
    g_edges = cv.Canny(g_abs_grad, histeresys_min_thres, histeresys_max_thres , None, aperture_size)
    b_edges = cv.Canny(b_abs_grad, histeresys_min_thres, histeresys_max_thres, None, aperture_size)
    r_edges_erode = morphology_filters(r_edges)
    g_edges_erode = morphology_filters(g_edges)
    b_edges_erode = morphology_filters(b_edges)
    return r_edges_erode, g_edges_erode, b_edges_erode, r_edges, g_edges, b_edges

r_edges_erode, g_edges_erode, b_edges_erode, r_edges, g_edges, b_edges = multi_edges(r_abs_grad,
                                                          g_abs_grad,
                                                          b_abs_grad)
    
# Ejecutamos Hough
def get_rgbchannel_lines(r_edges_erode, g_edges_erode, b_edges_erode, columns):
    
    min_long = int(columns*0.30)
    min_gap = int(columns*0.03)
    r_lines = cv.HoughLinesP(r_edges_erode, 1, np.pi/180, 60, 30, min_long, min_gap)
    g_lines = cv.HoughLinesP(g_edges_erode, 1, np.pi/180, 60, 30, min_long, min_gap)
    b_lines = cv.HoughLinesP(b_edges_erode, 1, np.pi/180, 60, 30, min_long, min_gap)
    return r_lines, g_lines, b_lines
r_lines, g_lines, b_lines = get_rgbchannel_lines(r_edges_erode,
                                                 g_edges_erode,
                                                 b_edges_erode,
                                                 columns)

# Dibujamos las líneas resultantes sobre una copia de la imagen original
# dst = img.copy()
# dst2 = img.copy()
# dst3 = img.copy()

if r_lines is not None:
    for i in range(0, len(r_lines)):
        l = r_lines[i][0]
        cv.line(dst, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv.LINE_AA)
if g_lines is not None:
    for i in range(0, len(g_lines)):
        l = g_lines[i][0]
        cv.line(dst, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv.LINE_AA)
if b_lines is not None:
    for i in range(0, len(b_lines)):
        l = b_lines[i][0]
        cv.line(dst, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv.LINE_AA)

# all_lines = [*r_lines, *g_lines, *b_lines]

# print(all_lines)

#CREAMOS VECTOR DE FEATURES
h, s, v = cv.split(img_hsv)

# img_grad = cv.merge((h, s, v, b_abs_grad))
img_grad = cv.merge((h, s, v, r_abs_grad, g_abs_grad, b_abs_grad))

# # print(np.shape(img_grad))
# imshow('img_grad', img_grad)

# # ## Uso de k-means
def k_means_for_feature_vector(img_grad):
    # Z = img_grad.reshape((-1,4))
    Z = img_grad.reshape((-1,6))

    # # convert to np.float32
    Z = np.float32(Z)
    # # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    # # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img_grad.shape))

    # r,g,b,grad1 = cv.split(res2)
    r,g,b,grad1, grad2, grad3 = cv.split(res2)

    res2 = merge((r,g,b))
    return res2

def k_means_for_printed_lines():

# # cv.imshow('imgdeg',img_grad)  
# cv.imshow('dst',res2)  
      
   

                ######------- IMSHOWS--------- #######
     
# imshow('resultado preprocesamiento', result_prep)
# imshow('entrada', img)
# imshow('post-filtrado bilateral y meanshift', img_rgb)  
# imshow('grad con valor absoluto', abs_grad)
# # imshow('grad con valor de raiz', sqr_grad) 
# imshow('img_gray', img_gray)
 
# # imshow('res2', res2) 
# # imshow('resultado preprocesamiento CLAHE', result2)

# imshow('gradientes de red channel', r_abs_grad)
# imshow('gradientes de green channel', g_abs_grad)
# imshow('gradientes de blue channel', b_abs_grad)
# imshow('bordes_erode r', r_edges_erode)
# imshow('bordes_erode g', g_edges_erode)
# imshow('bordes_erode b', b_edges_erode)
# imshow('bordes r', r_edges)
# imshow('bordes g', g_edges)
# imshow('bordes b', b_edges)
imshow('Lineas r', dst)
# imshow('Lineas g', dst2)
# imshow('Lineas b', dst3)
# imshow('Lineas_erode', edges_erode)



cv.waitKey(0)
# # print(np.shape(l))
cv.destroyAllWindows()


