import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, feature
import os
import cv2
import argparse
# Cargar la imagen
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "nombre de la imagen")

args = vars(ap.parse_args())
path = os.path.join('images', 'img{}.png'.format(args['image']))
n = args['image']
save_path = os.path.join('results', '{}.png'.format(n))
save_path2 = os.path.join('results', 'colour{}.png'.format(n))
img = cv2.imread(path)

def create_gaborfilter():
    # This function is designed to produce a set of GaborFilters 
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree
     
    filters = []
    num_filters = 16
    ksize = 50  # The local area to evaluate
    sigma = 3  # Larger Values produce more edges
    lambd = 10.0
    gamma = 0.5
    psi = 0  # Offset value - lower generates cleaner results
    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)
    return filters
def preprocessing(img):
    # img2 = cv2.cv2tColor(img, cv2.COLOR_BGR2RGB)
    
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # l_blur_st = time.time()
    rows, columns, channel  = img_lab.shape
    
    l, a, b = cv2.split(img_lab)

    # l_blur_st = time.time()
    l_blur = cv2.medianBlur(l, 5)
    # l_blur_end = time.time()
    # cielab_st = time.time()
    # for row in range(rows):
        # for column in range(columns):                    
    img_lab[:, :, 0] = (1.5*img_lab[:, :, 0]) - (0.5*l_blur[:, :]) 
            
    # result = cv2.cv2tColor(img_lab, cv2.COLOR_LAB2RGB)
    # result_gr = cv2.cv2tColor(img_lab, cv2.COLOR_RGB2GRAY)
    
    # cielab_end = time.time()
    # create a CLAHE object (Arguments are optional).
    # clahe_st = time.time()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    img_lab[...,0] = clahe.apply(img_lab[...,0])
    # clahe_end = time.time()

    # blur_time = l_blur_end - l_blur_st
    # cielab_time = cielab_end - cielab_st
    # clahe_time = clahe_end - clahe_st
    # print('BLUR time ' '{}: '.format(n), blur_time, 'seconds - ', _time_per)
    # print('CIELAB time ' '{}: '.format(n), cielab_time, 'seconds - ', _time_per)
    # print('CLAHE time ' '{}: '.format(n), clahe_time, 'seconds - ', _time_per)
    result_prep = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    result_prep = white_balance(result_prep)
    return result_prep
def apply_filter(img, filters):
# This general function is designed to apply filters to our image
     
    # First create a numpy array the same size as our input image
    newimage = np.zeros_like(img)
     
    # Starting with a blank image, we loop through the images and apply our Gabor Filter
    # On each iteration, we take the highest value (super impose), until we have the max value across all filters
    # The final image is returned
    depth = -1 # remain depth same as original image
     
    for kern in filters:  # Loop through the kernels in our GaborFilter
        image_filter = cv2.filter2D(img, depth, kern)  #Apply filter to image
         
        # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
        np.maximum(newimage, image_filter, newimage)
    return newimage, image_filter
def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result 
# We create our gabor filters, and then apply them to our image

gfilters = create_gaborfilter()
img_prep = preprocessing(img)
image_g, filter = apply_filter(img_prep, gfilters)
min_interval = 120
max_interval = 250
image_edge_g = cv2.Canny(image_g,min_interval,max_interval)
# # Definir los parámetros para los filtros de Gabor
# frequencies = [0.1, 0.2, 0.3] # Frecuencias de los filtros
# thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4] # Orientaciones de los filtros
# sigmas = [1, 3, 5] # Desviación estándar de las funciones gaussianas

# # Aplicar los filtros de Gabor a la imagen
# gabor_responses = np.zeros((img.shape[0], img.shape[1], len(frequencies), len(thetas), len(sigmas)))

# for i, freq in enumerate(frequencies):
#     for j, theta in enumerate(thetas):
#         for k, sigma in enumerate(sigmas):
#             gabor_filter = filters.gabor_kernel(freq, theta=theta, sigma_x=sigma, sigma_y=sigma)
#             gabor_responses[:, :, i, j, k] = feature.canny(filters.convolve(img, gabor_filter))
            

cv2.imshow('Gabor', image_edge_g)
# cv2.imshow('Gabor filter', filter)
cv2.waitKey(0)
cv2.destroyAllWindows()