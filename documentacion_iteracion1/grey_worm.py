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
from skimage.draw import line
from matplotlib import pyplot as plt



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "nombre de la imagen")
# ap.add_argument("-n", "--n_image", required=True, help= "Número de imagen")
args = vars(ap.parse_args())
path = os.path.join('images', 'img{}.png'.format(args['image']))
# n = args['n_image']
# save_path = os.path.join('results', '{}.png'.format(n))
# save_path2 = os.path.join('results', 'colour{}.png'.format(n))
img = cv.imread(path)
print('ala')
def grey_world(img):
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        # #Gradientes
    ddepth = cv.CV_16S

    ksize = 3    
    r_grad_x = cv.Sobel(img_gray, ddepth, 1, 0, ksize=ksize, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    r_grad_y = cv.Sobel(img_gray, ddepth, 0, 1, ksize=ksize, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        
    r_abs_grad_x = cv.convertScaleAbs(r_grad_x)
    r_abs_grad_y = cv.convertScaleAbs(r_grad_y)

 

    #TODO: IMPLEMENTAR CALCULO DE SOBEL CON RAIZ CUADRADA   
    # sqr_grad = ((grad_x ** 2) + (grad_y ** 2))**0,5

    abs_grad = cv.addWeighted(r_abs_grad_x, 0.5, r_abs_grad_y, 0.5, 0)
    return abs_grad
    
res = grey_world(img)    
    
cv.imshow('ala', img)
cv.waitKey(0)
cv.destroyAllWindows()