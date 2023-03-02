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




# Read the image
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True, help = "nombre de la imagen")
ap.add_argument("-n", "--n_image", required=True, help= "Número de imagen")

args = vars(ap.parse_args())
n = args['n_image']
path = os.path.join('images', 'img{}.png'.format(n))
save_path = os.path.join('ideal_output', '{}.png'.format(n))


img = cv.imread(path)

# Remember -> OpenCV stores things in BGR order
lowerBound = (0, 0, 255);
upperBound = (0, 0, 255);

# this gives you the mask for those in the ranges you specified,
# but you want the inverse, so we'll add bitwise_not...
final_mask = cv.inRange(img, lowerBound, upperBound, 1);
erosion_type = cv.MORPH_RECT
    
    
    
    
element = cv.getStructuringElement(erosion_type, (2,2))
element2 = cv.getStructuringElement(erosion_type, (2,2))
final_mask = cv.dilate(final_mask, element)  

cv.imwrite(save_path, img=final_mask)



