#!/usr/bin/env python3
from ast import Pass
from logging import info
from typing_extensions import Self
from unittest import skip
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Empty
from cv_bridge import CvBridge 
import numpy as np
# from nav_msgs.msg import Odometry
# from tf.transformations import euler_from_quaternion, quaternion_from_euler
# from geometry_msgs.msg import Point, Twist
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
# from scipy.spatial.distance import pdist
# from scipy.cluster.hierarchy import ward, fcluster


def preprocessing(img):
    # img2 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    rows, columns, channel  = img_lab.shape
    l, a, b = cv.split(img_lab)

    l_blur = cv.medianBlur(l, 5)

    #SLOW
    # for row in range(rows):
    #     for column in range(columns):
    #         l_in=  img_lab[row, column, 0]
    #         l_out = (1.5*l_in) - (0.5*l_blur[row, column]) 
    #         img_lab[row, column, 0] = l_out
    
    #FAST
    img_lab[:, :, 0] = (1.5*img_lab[:, :, 0]) - (0.5*l_blur[:, :]) 
    
    result = cv.cvtColor(img_lab, cv.COLOR_LAB2RGB)
    result_gr = cv.cvtColor(img_lab, cv.COLOR_RGB2GRAY)
    # create a CLAHE object (Arguments are optional).
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    img_lab[...,0] = clahe.apply(img_lab[...,0])



    result_prep = cv.cvtColor(img_lab, cv.COLOR_LAB2RGB)
    return result_prep

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

def multi_edges(r_abs_grad, g_abs_grad, b_abs_grad):
    aperture_size = 3
    histeresys_min_thres = 150
    histeresys_max_thres = 180
    gfilters = create_gaborfilter()
    gabor_r, _ = apply_filter(r_abs_grad, gfilters)
    gabor_g, _ = apply_filter(g_abs_grad, gfilters)
    gabor_b, _ = apply_filter(b_abs_grad, gfilters)
    r_edges = cv.Canny(gabor_r, histeresys_min_thres, histeresys_max_thres, None, aperture_size)
    g_edges = cv.Canny(gabor_g, histeresys_min_thres, histeresys_max_thres , None, aperture_size)
    b_edges = cv.Canny(gabor_b, histeresys_min_thres, histeresys_max_thres, None, aperture_size)
    r_edges_erode = morphology_filters(r_edges)
    g_edges_erode = morphology_filters(g_edges)
    b_edges_erode = morphology_filters(b_edges)
    return r_edges_erode, g_edges_erode, b_edges_erode, r_edges, g_edges, b_edges
def get_rgbchannel_lines(r_edges_erode, g_edges_erode, b_edges_erode, columns):
    
    min_long = int(columns*0.35)
    min_gap = int(columns*0.04)
    r_lines = cv.HoughLinesP(r_edges_erode, 1, np.pi/180, 70, 40, min_long, min_gap)
    g_lines = cv.HoughLinesP(g_edges_erode, 1, np.pi/180, 70, 40, min_long, min_gap)
    b_lines = cv.HoughLinesP(b_edges_erode, 1, np.pi/180, 70, 40, min_long, min_gap)
                    
    return r_lines, g_lines, b_lines

def get_rgbchannel_lines2(r_edges_erode, g_edges_erode, b_edges_erode, columns):
    
    min_long = int(columns*0.45)
    min_gap = int(columns*0.05)
    r_lines = cv.HoughLinesP(r_edges_erode, 1, np.pi/180, 70, 40, min_long, min_gap)
    g_lines = cv.HoughLinesP(g_edges_erode, 1, np.pi/180, 70, 40, min_long, min_gap)
    b_lines = cv.HoughLinesP(b_edges_erode, 1, np.pi/180, 70, 40, min_long, min_gap)
                    
    return r_lines, g_lines, b_lines
def filter_masks_by_area(masks, area_img, min_area_percent, max_area_percent):
    """
    Filter binary masks by area in percentage terms.

    Args:
        masks (list): List of binary masks.
        area_img (int): Total number of pixels in the image.
        min_area_percent (float): Minimum percentage of area a mask must have to be kept.
        max_area_percent (float): Maximum percentage of area a mask can have to be kept.

    Returns:
        filtered_masks (list): List of binary masks that meet the area criteria.
    """
    filtered_masks = []
    i=0
    for mask in masks:
        
        area = np.sum(mask == 255)

        area_percent = area / area_img * 100.0
        print("area de la máscara ",i,": ", area_percent)
        if min_area_percent <= area_percent <= max_area_percent:
            filtered_masks.append(mask)
        i=i+1 
    return filtered_masks

def print_lines_and_get_all_lines(r_lines, g_lines, b_lines, printed_lines_img):
    # Dibujamos las líneas resultantes sobre una copia de la imagen original
    # dst = img.copy()
    # dst2 = img.copy()
    # dst3 = img.copy()
    # print('que pasa')
    rl = False
    gl = False
    bl = False
    if r_lines is not None:        
        rl= True
        for i in range(0, len(r_lines)):
            
            l = r_lines[i][0]
            cv.line(printed_lines_img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv.LINE_AA)
    if g_lines is not None:
        gl = True
        for i in range(0, len(g_lines)):
            
            l = g_lines[i][0]
            cv.line(printed_lines_img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv.LINE_AA)
    if b_lines is not None:
        bl = True
        # all_lines = b_lines
        for i in range(0, len(b_lines)):
            l = b_lines[i][0]
            cv.line(printed_lines_img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv.LINE_AA)

    if rl and gl and bl:
        all_lines = np.concatenate((r_lines, g_lines,b_lines), axis=0)
    elif rl and gl:
        all_lines = np.concatenate((r_lines, g_lines), axis=0)
    elif rl and bl:
        all_lines = np.concatenate((r_lines, b_lines), axis=0)
    elif gl and bl:    
        all_lines = np.concatenate((g_lines, b_lines), axis=0)      
    elif rl:
        all_lines = r_lines
    elif gl:
        all_lines = g_lines
    elif bl:        
        all_lines = b_lines
    else: 
        all_lines = None
    return all_lines, printed_lines_img    

def findparallel(all_lines, columns):    
    # print(all_lines)
    if all_lines is not None:
        parallel_lines = []
        dist_threshold = columns*0.25
        for i in range(len(all_lines)):
            for j in range(len(all_lines)):
                if (i == j):continue
                # print('linea 1 ', all_lines[i][0], ' - ', 'linea 2', all_lines[j][0])
                
                xa_1 = all_lines[i][0][0] 
                ya_1 = all_lines[i][0][1] 
                
                xb_1 = all_lines[i][0][2] 
                yb_1 = all_lines[i][0][3] 
                
                # print((yb_1-ya_1))
                # print(xb_1-xa_1)
                m1 = (yb_1-ya_1)/(xb_1-xa_1)
                # b1 = ya_1 - m1*xa_1
                
                
                xa_2 = all_lines[j][0][0] 
                ya_2 = all_lines[j][0][1] 
                
                xb_2 = all_lines[j][0][2] 
                yb_2 = all_lines[j][0][3] 
                
                m2 = (yb_2-ya_2)/(xb_2-xa_2)
                ## Paralelismo basado en pendiente ##
                # b2 = ya_2 - m2*xa_2
                # m_dif = m1-m2
                # if m1 >= 60:
                #     slopediff_threshold = 60
                # if m1 <60:
                #     slopediff_threshold = 3    
                # print(xa_1, ya_1, xb_1, yb_1)
                # print(m1,' - ', m2, '=', m_dif)
                
                ## Paralelismo basado en angulo respecto a la horizontal ##
                ang1 = math.degrees(math.atan(m1))
                ang2 = math.degrees(math.atan(m2))
                ang_dif = abs(ang1-ang2)
                ang_dif_threshold = 10
                # print(ang1, ' ', ang2, ' ', ang_dif)
                
                ## Distancia entre paralelas:
                dist_x = abs(xa_1-xa_2)
                dist_y = abs(ya_1-ya_2)
                
                
                if ( ang_dif <= ang_dif_threshold and dist_y < dist_threshold and dist_x < dist_threshold ):          
                #You've found a parallel line!
                    # print('cumplen la condicion')
                    # parallel_lines.append(all_lines[i])
                    parallel_lines.append(all_lines[j])
                    
                    
                    # print(parallel_lines)             
                    # parallel_lines = np.concatenate((parallel_lines,all_lines[i]), axis=0)
                    # parallel_lines = np.concatenate((parallel_lines,all_lines[j]), axis=0)

        # parallel_lines = np.unique(parallel_lines)
    else:
        parallel_lines = None    
    return parallel_lines

def findparallel2(all_lines, columns):    
    # print(all_lines)
    if all_lines is not None:
        parallel_lines = []
        dist_threshold = columns*0.25
        biggest_line_index, biggest_line_angle, _ = find_biggest_line(all_lines=all_lines)
        l = len(all_lines)
        
        if l==1:
            parallel_lines.append(all_lines[0])
        else:
            parallel_lines.append(all_lines[biggest_line_index])
            for j in range(l):
                if (j == biggest_line_index):continue
                # print('linea 1 ', all_lines[i][0], ' - ', 'linea 2', all_lines[j][0])
                
                xa_1 = all_lines[biggest_line_index][0][0] 
                ya_1 = all_lines[biggest_line_index][0][1] 
                
                xb_1 = all_lines[biggest_line_index][0][2] 
                yb_1 = all_lines[biggest_line_index][0][3] 
                
                # print((yb_1-ya_1))
                # print(xb_1-xa_1)
                # m1 = (yb_1-ya_1)/(xb_1-xa_1)
                # b1 = ya_1 - m1*xa_1
                
                
                xa_2 = all_lines[j][0][0] 
                ya_2 = all_lines[j][0][1] 
                
                xb_2 = all_lines[j][0][2] 
                yb_2 = all_lines[j][0][3] 
                
                m2 = (yb_2-ya_2)/(xb_2-xa_2)
                ## Paralelismo basado en pendiente ##
                # b2 = ya_2 - m2*xa_2
                # m_dif = m1-m2
                # if m1 >= 60:
                #     slopediff_threshold = 60
                # if m1 <60:
                #     slopediff_threshold = 3    
                # print(xa_1, ya_1, xb_1, yb_1)
                # print(m1,' - ', m2, '=', m_dif)
                
                ## Paralelismo basado en angulo respecto a la horizontal ##
                # ang1 = math.degrees(math.atan(m1))
                ang2 = math.degrees(math.atan(m2))
                ang_dif = abs(biggest_line_angle-ang2)
                ang_dif_threshold = 10
                # print(ang1, ' ', ang2, ' ', ang_dif)
                
                ## Distancia entre paralelas:
                dist_x = abs(xa_1-xa_2)
                dist_y = abs(ya_1-ya_2)
                
                
                if ( ang_dif <= ang_dif_threshold and dist_y < dist_threshold and dist_x < dist_threshold ):          
                #You've found a parallel line!
                    # print('cumplen la condicion')
                    # parallel_lines.append(all_lines[i])
                    parallel_lines.append(all_lines[j])
                
                
                # print(parallel_lines)             
                # parallel_lines = np.concatenate((parallel_lines,all_lines[i]), axis=0)
                # parallel_lines = np.concatenate((parallel_lines,all_lines[j]), axis=0)

        # parallel_lines = np.unique(parallel_lines)
    else:
        parallel_lines = None    
    return parallel_lines, biggest_line_index

def find_biggest_line(all_lines):
    long = 0
    max_long = 0
    ang = 0
    index = 0
    for i in range(len(all_lines)):
    
        
        # print('linea 1 ', all_lines[i][0], ' - ', 'linea 2', all_lines[j][0])
        
        xa_1 = all_lines[i][0][0] 
        
        ya_1 = all_lines[i][0][1] 
        
        xb_1 = all_lines[i][0][2] 
        yb_1 = all_lines[i][0][3] 
        
        long = math.sqrt(((xb_1-xa_1)**2)+((yb_1-ya_1)**2))
        long = int(long)
        # print(long, '-------', i)
        if long > max_long:
            max_long = long
            m = (yb_1-ya_1)/(xb_1-xa_1)         
            index = i
            ang = math.degrees(math.atan(m)) 
        
        # print((yb_1-ya_1))
        # print(xb_1-xa_1)
    return index, ang, max_long   

def white_balance(img):
    result = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv.cvtColor(result, cv.COLOR_LAB2BGR)
    return result    

def line_filtering(img, all_lines, columns):
    n=0
    if len(all_lines)==1:
            l = all_lines[0][0]
                    # print(l[0])
            cv.line(img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 2, cv.LINE_AA)
    else:    
        while (n < 6) and (len(all_lines)>1):
            # parallel_lines = findparallel(all_lines, columns)
            parallel_lines, big_l_indx = findparallel2(all_lines, columns)
            # del all_lines[big_l_indx]
            # print(all_lines)
            all_lines = np.delete(all_lines, big_l_indx, axis=0)
            # print('juan')
            # print(big_l_indx)
            # print(all_lines)
            
            if parallel_lines is not None:        
                
                for i in range(0, len(parallel_lines)): 
                        
                    l = parallel_lines[i][0]
                    # print(l[0])
                    cv.line(img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 2, cv.LINE_AA)
            n=n+1
    return img    
def smallest_mask(masks):
    areas = [np.sum(mask) for mask in masks]
    return np.argmin(areas)
def morphology_filters2(img_bin):
        # Forma del filtro
    erosion_type = cv.MORPH_RECT
    erosion_type2 = cv.MORPH_ELLIPSE
    # El último parámetro es el tamaño del filtro, en este caso 5x5
    # element = cv.getStructuringElement(erosion_type, (4,4)) 
    element2 = cv.getStructuringElement(erosion_type2, (1,1)) 
    element3 = cv.getStructuringElement(erosion_type2, (5,5))
    element4 = cv.getStructuringElement(erosion_type, (5,5)) 
    # dst = cv.erode(dst,element2)
    # dst = cv.morphologyEx(img_meanshift, cv.MORPH_CLOSE, element2)
    # dst2 = cv.morphologyEx(dst, cv.MORPH_CLOSE, element2)
    edges_erode = cv.erode(img_bin.astype(np.uint8), element2)
    edges_erode = cv.erode(edges_erode, element2)
    edges_erode = cv.erode(edges_erode, element2)
    edges_erode = cv.erode(edges_erode, element2)
    edges_erode = cv.erode(edges_erode, element2)
    edges_erode = cv.erode(edges_erode, element2)
    # edges_erode = cv.dilate(edges_erode, element4)
    
    
    # edges_erode = cv.dilate(edges_erode, element4)
    # edges_erode = cv.erode(edges_erode, element3)
    
   
    return edges_erode

def largest_mask(masks):
    areas = [np.sum(mask) for mask in masks]
    return np.argmax(areas)

def create_gaborfilter():
    # This function is designed to produce a set of GaborFilters 
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree
     
    filters = []
    num_filters = 16
    ksize = 50  # The local area to evaluate
    sigma = 2.2  # Larger Values produce more edges
    lambd = 10.0
    gamma = 0.5
    psi = 0  # Offset value - lower generates cleaner results
    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv.CV_64F)
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)
    return filters

def apply_filter(img, filters):
# This general function is designed to apply filters to our image
     
    # First create a numpy array the same size as our input image
    newimage = np.zeros_like(img)
     
    # Starting with a blank image, we loop through the images and apply our Gabor Filter
    # On each iteration, we take the highest value (super impose), until we have the max value across all filters
    # The final image is returned
    depth = -1 # remain depth same as original image
     
    for kern in filters:  # Loop through the kernels in our GaborFilter
        image_filter = cv.filter2D(img, depth, kern)  #Apply filter to image
         
        # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
        np.maximum(newimage, image_filter, newimage)
    return newimage, image_filter


def hsv_gradient_segmentation(img, num_clusters=5):
    # Convert image to grayscale
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Calculate gradients
    # sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    # sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    # gradient = np.sqrt(np.square(sobelx) + np.square(sobely))

    r_abs_grad, g_abs_grad, b_abs_grad = splitted_sobels(img)
    r_edges_erode, g_edges_erode, b_edges_erode, r_edges, g_edges, b_edges = multi_edges(r_abs_grad,
                                                        g_abs_grad,
                                                        b_abs_grad)
    # Convert image from BGR to HSV
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Reshape the image to a 2D array of pixels
    pixels = hsv_img.reshape((-1, 3))
    h,w, _ = hsv_img.shape
    area_img = h*w
    # Compute feature vector using gradient and HSV values
    feature_vector = np.concatenate((pixels, r_edges_erode.flatten().reshape((-1, 1))), axis=1)
    feature_vector = np.concatenate((feature_vector, g_edges_erode.flatten().reshape((-1, 1))), axis=1)
    feature_vector = np.concatenate((feature_vector, b_edges_erode.flatten().reshape((-1, 1))), axis=1)

    # Convert feature vector values to float32 for clustering
    feature_vector = np.float32(feature_vector)
    # print(feature_vector.shape)
    # Define criteria and apply k-means clustering
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centers = cv.kmeans(feature_vector, num_clusters, None, criteria, 10, cv.KMEANS_PP_CENTERS)

    # Reshape the labels to match the original image shape
    labels = labels.reshape(img.shape[:2])

    # Create a mask for each cluster
    masks = []
    for i in range(num_clusters):
        mask = np.zeros_like(labels)
        mask[labels == i] = 255
        masks.append(mask)
    filtered_masks = filter_masks_by_area(masks,area_img, 4, 25)
    # print(len(filtered_masks))
    # # Combine masks to create final segmentation mask
    segmentation_mask = np.zeros_like(labels)
    # segmentation_mask2 = np.zeros_like(labels)
    # segmentation_mask3 = np.zeros_like(labels)
    # for mask in masks:
    #     segmentation_mask = cv.bitwise_or(segmentation_mask, mask)
    # print(masks)
    # biggest_area_index = largest_mask(masks)
    # print(biggest_area_index)
    # smallest_area_index = smallest_mask(masks)
    # print(smallest_area_index)
    # second_smallest_index = second_smallest_mask(masks)
    # print(second_smallest_index)
    # candidates = []  
    # for i in range(num_clusters):
    #     if i== biggest_area_index:
    #         continue
    #     if i== smallest_area_index:
    #         continue
    #     if i== second_smallest_index:
    #         continue
    #     # else:
    #     #     if not c1_check: 
    #     #         c1 = masks[i]
    #     #         c1 = morphology_filters(c1)
    #     #         c1_check = True
    #     #         continue
    #     #     c2 = masks[i]    
    #     #     c1 = morphology_filters(c2)
    #     #     segmentation_mask = cv.bitwise_and(c1, c2)
    #     # segmentation_mask = cv.bitwise_or(segmentation_mask, masks[i])
    #     candidates.append(masks[i])
    for i in range(len(filtered_masks)):
        segmentation_mask = cv.bitwise_or(segmentation_mask, filtered_masks[i])
    # print(candidates)
    # segmentation_mask = get_best_rectangular_contour(candidates[0],candidates[1])
    # segmentation_mask = compare_masks_rectangularity(candidates[0], candidates[1])
    
    # segmentation_masks = [None] * num_clusters      
    # for i in range(num_clusters):
        
    #     # segmentation_mask3 = cv.bitwise_or(segmentation_mask3, masks[2])        
    #     segmentation_masks[i] = np.zeros_like(labels)
    #     # segmentation_mask2 = np.zeros_like(labels)
    #     # segmentation_mask3 = np.zeros_like(labels)
    #     # segmentation_mask4 = np.zeros_like(labels)
    #     segmentation_masks[i] = cv.bitwise_or(segmentation_masks[i], masks[i])
    #     # segmentation_mask2 = cv.bitwise_or(segmentation_mask2, masks[1])
    #     # segmentation_mask3 = cv.bitwise_or(segmentation_mask3, masks[2])
    #     # segmentation_mask4 = cv.bitwise_or(segmentation_mask4, masks[3])
    #     # Convert mask to uint8 data type
    #     segmentation_masks[i] = segmentation_masks[i].astype(np.uint8)
    #     # segmentation_mask1 = segmentation_mask1.astype(np.uint8)
    #     # segmentation_mask2 = segmentation_mask2.astype(np.uint8)
    #     # segmentation_mask3 = segmentation_mask3.astype(np.uint8)
    #     cv.imshow('cluster {}'.format(i), segmentation_masks[i])    
    
    
    
    # segmentation_mask4 = segmentation_mask4.astype(np.uint8)
    # segmentation_mask3 = segmentation_mask3.astype(np.uint8)
    
    
    
    
    # cv.imshow('cluster4', segmentation_mask4)
    # cv.waitKey(0)
    # cv.destroyAllWindows()    
    
    # Apply mask to original image
    # segmented_img = cv.bitwise_and(img, img, mask=segmentation_mask1)
    # img[segmentation_mask==255] = (0, 0, 255)
    # Convert segmented image back to BGR color space
    # segmented_img = cv.cvtColor(segmented_img, cv.COLOR_HSV2BGR)

    return segmentation_mask

class Nodo(object):
    
    def __init__(self):
        # Params
        self.r = rospy.Rate(1)
        self.image = None
        self.br = CvBridge()
        # self.cmd = Twist()
        
        
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(1)
        self.image_rgb = None
        
        # Publishers
       
        # TODO Saber en que topic se publica 
        # self.pub = rospy.Publisher('/uone/masks/structure_detector', Image, queue_size=5)
        self.pub = rospy.Publisher('/uone/masks/structure_detector', Image, queue_size=5)
        

        # TODO Saber cual es el topic que publica la imagen de la camara
        self.sub = rospy.Subscriber(
            "/uone/stereo/left/image", Image, self.callback)
        # self.sub = rospy.Subscriber(
        #     "bslmaris/central_image_raw", Image, self.callback)
        # self.sub2 = rospy.Subscriber("/camera/rgb/image_raw",Image,self.callback2)
        
        #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
        

    
    #Leemos imagen de la camara uOne
    def callback(self, msg):
        rospy.loginfo('Image received...')
        self.image = self.br.imgmsg_to_cv2(msg, "bgr8")

    
    def start(self):
        rospy.loginfo('Weno dia estar')    
            
        while True:
            if rospy.is_shutdown():
                break  
            if self.image is not None:
                
                # rospy.loginfo('Weno dia if')    
                #TODO: AQUI VA EL ALGORITMO DE VISIÓN

        #-------------------------------------  PRE-PROCESSING  ----------------------
                img=self.image


                result_prep = preprocessing(img)
                wb= white_balance(result_prep)
                # img_rgb = result_prep
                # img_hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)

            #--------------Clusterización -----------------------------------------#


                rows, columns, channel  = result_prep.shape
                # features = np.zeros(shape=(rows,columns, 6))


                img_bilat = cv.bilateralFilter(wb, 20, sigmaColor=80, sigmaSpace=80)
                # img_meanshift = cv.pyrMeanShiftFiltering(img_bilat, sp=45, sr=25, maxLevel=2)

                # img_meanshift = cv.pyrMeanShiftFiltering(img_bilat, sp=45, sr=25, maxLevel=4)
                final_mask = hsv_gradient_segmentation(img_bilat)
                # clusters_res = cluster_classification(masks)
                # print(len(clusters_res))
                img_final = img.copy()
                img_final[final_mask==255] = (0, 0, 255)
                self.pub.publish(self.br.cv2_to_imgmsg(img_final))   
        #     else:
        #         break    
                
                
                    
                               
                      
            
                                                                    

    


if __name__ == '__main__':
    try:
        rospy.init_node("pipe_detector", anonymous=True)
        my_node = Nodo()
        my_node.start()
    except rospy.ROSInterruptException:
        pass