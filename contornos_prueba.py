from __future__ import print_function
import cv as cv
import numpy as np
import argparse
import random as rng
import os

# Load source image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "nombre de la imagen")

args = vars(ap.parse_args())
path = os.path.join('images', args['image'])

image= cv.imread(path)
original_image= image

gray= cv.cvtColor(image,cv.COLOR_BGR2GRAY)

edges= cv.Canny(gray, 50,200)

contours, hierarchy= cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)


cv.destroyAllWindows()


def get_contour_areas(contours):

    all_areas= []

    for cnt in contours:
        area= cv.contourArea(cnt)
        all_areas.append(area)

    return all_areas


sorted_contours= sorted(contours, key=cv.contourArea, reverse= True)


largest_item= sorted_contours[0]

cv.drawContours(original_image, largest_item, -1, (255,0,0),10)
cv.waitKey(0)
cv.imshow('Largest Object', original_image)


cv.waitKey(0)
cv.destroyAllWindows()