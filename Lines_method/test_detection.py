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
import time


# Read the image
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True, help = "nombre de la imagen")
# ap.add_argument("-n", "--n_image", required=True, help= "Número de imagen")
# args = vars(ap.parse_args())
# path = os.path.join('images', args['image'])
# n = args['n_image']
# save_path = os.path.join('test', 'bilatTest_{}.png'.format(n))
# save_path2 = os.path.join('test', 'bilatTest_grads_{}.png'.format(n))
# Read the image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "nombre de la imagen")
ap.add_argument("-n", "--n_image", required=True, help= "Número de imagen")
args = vars(ap.parse_args())
path = os.path.join('images', 'img{}.png'.format(args['image']))
n = args['n_image']
save_path = os.path.join('results', '{}.png'.format(n))
save_path2 = os.path.join('results', 'colour{}.png'.format(n))
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
def preprocessing(img):
    # img2 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    img_lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    l_blur_st = time.time()
    rows, columns, channel  = img_lab.shape
    
    l, a, b = cv.split(img_lab)

    # l_blur_st = time.time()
    l_blur = cv.medianBlur(l, 5)
    l_blur_end = time.time()
    cielab_st = time.time()
    # for row in range(rows):
        # for column in range(columns):                    
    img_lab[:, :, 0] = (1.5*img_lab[:, :, 0]) - (0.5*l_blur[:, :]) 
            
    # result = cv.cvtColor(img_lab, cv.COLOR_LAB2RGB)
    # result_gr = cv.cvtColor(img_lab, cv.COLOR_RGB2GRAY)
    
    cielab_end = time.time()
    # create a CLAHE object (Arguments are optional).
    clahe_st = time.time()
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    img_lab[...,0] = clahe.apply(img_lab[...,0])
    clahe_end = time.time()

    # blur_time = l_blur_end - l_blur_st
    # cielab_time = cielab_end - cielab_st
    # clahe_time = clahe_end - clahe_st
    # print('BLUR time ' '{}: '.format(n), blur_time, 'seconds - ', _time_per)
    # print('CIELAB time ' '{}: '.format(n), cielab_time, 'seconds - ', _time_per)
    # print('CLAHE time ' '{}: '.format(n), clahe_time, 'seconds - ', _time_per)
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

    ksize = 3    
    r_grad_x = cv.Sobel(r_img, ddepth, 1, 0, ksize=ksize, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    r_grad_y = cv.Sobel(r_img, ddepth, 0, 1, ksize=ksize, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        
    r_abs_grad_x = cv.convertScaleAbs(r_grad_x)
    r_abs_grad_y = cv.convertScaleAbs(r_grad_y)

    g_grad_x = cv.Sobel(g_img, ddepth, 1, 0, ksize=ksize, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    g_grad_y = cv.Sobel(g_img, ddepth, 0, 1, ksize=ksize, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        
    g_abs_grad_x = cv.convertScaleAbs(g_grad_x)
    g_abs_grad_y = cv.convertScaleAbs(g_grad_y)

    b_grad_x = cv.Sobel(b_img, ddepth, 1, 0, ksize=ksize, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    b_grad_y = cv.Sobel(b_img, ddepth, 0, 1, ksize=ksize, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        
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
            cv.line(printed_lines_img, (l[0], l[1]), (l[2], l[3]), (0,255,0), 1, cv.LINE_AA)
    if b_lines is not None:
        bl = True
        # all_lines = b_lines
        for i in range(0, len(b_lines)):
            l = b_lines[i][0]
            cv.line(printed_lines_img, (l[0], l[1]), (l[2], l[3]), (255,0,0), 1, cv.LINE_AA)

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

def draw_mask(printed_lines_img_2, paralle_lines, columns):
    parallels = findparallel2(paralle_lines,columns)
    biggest_line_index, _, max_long = find_biggest_line(parallels)
    length = len(parallels)
    
    xa_1 = parallels[biggest_line_index][0][0] 
    ya_1 = parallels[biggest_line_index][0][1] 
    xb_1 = parallels[biggest_line_index][0][2] 
    yb_1 = parallels[biggest_line_index][0][3]     
    # print(length)
    if length == 1:
        cv.line(printed_lines_img_2, (xa_1, ya_1), (xb_1, yb_1), (0,0,255), 30, cv.LINE_AA)
    
        # cv.rectangle(printed_lines_img_2,(xa_1-20,ya_1-20),(xb_1+20,yb_1+20),(0,0,255),-3)
    
    else:
            # being start and end two points (x1,y1), (x2,y2)
        discrete_line = list(zip(*line(*(xa_1,ya_1), *(xb_1,yb_1))))
        for i in range(length):    
            if i == biggest_line_index:
                continue
            

            
            x = parallels[i][0][0]
            y = parallels[i][0][1]

            for j in range(len(discrete_line)):

                x2 , y2 = discrete_line[j]
                cv.line(printed_lines_img_2, (x, y), (x2, y2), (0,0,255), 2, cv.LINE_AA)
    
    return printed_lines_img_2

def draw_mask2(printed_lines_img_2, paralle_lines, columns):
        parallels = findparallel2(paralle_lines,columns)
        biggest_line_index, _, max_long = find_biggest_line(parallels)
        length = len(parallels)
        
        xa_1 = parallels[biggest_line_index][0][0] 
        ya_1 = parallels[biggest_line_index][0][1] 
        xb_1 = parallels[biggest_line_index][0][2] 
        yb_1 = parallels[biggest_line_index][0][3]     
        # print(length)
        if length == 1:
            cv.line(printed_lines_img_2, (xa_1, ya_1), (xb_1, yb_1), (0,0,255), 30, cv.LINE_AA)
        
            # cv.rectangle(printed_lines_img_2,(xa_1-20,ya_1-20),(xb_1+20,yb_1+20),(0,0,255),-3)
        
        else:
            del parallels[biggest_line_index]
            second_biggest_line_index, _, _ = find_biggest_line(parallels)
            # Polygon corner points coordinates
            pts = np.array([[xa_1, ya_1], [xb_1, yb_1],
                            [parallels[second_biggest_line_index][0][0], parallels[second_biggest_line_index][0][1]],
                            [parallels[second_biggest_line_index][0][2], parallels[second_biggest_line_index][0][3]]],
                        np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv.fillPoly(printed_lines_img, pts=pts, color=(0,255,255))
        return printed_lines_img_2    

def line_filtering(img, all_lines, columns):
    if all_lines is not None:
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

def create_gaborfilter():
    # This function is designed to produce a set of GaborFilters 
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree
     
    filters = []
    num_filters = 16
    ksize = 50  # The local area to evaluate
    sigma = 3.5  # Larger Values produce more edges
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



if __name__== '__main__':
    #-------------------------------------  PRE-PROCESSING  ----------------------

    start = time.time()
    
    preprocessing_st = time.time()
    result_prep = preprocessing(img)
    preprocessing_end = time.time()
    
    wb_st = time.time()
    wb= white_balance(result_prep)
    wb_end = time.time()
    #--------------Clusterización -----------------------------------------#


    rows, columns, channel  = result_prep.shape
    # features = np.zeros(shape=(rows,columns, 6))
    bilat_st = time.time()
    
    #  = cv.GaussianBlur(wb,(3,3),cv.BORDER_WRAP)
    img_bilat = cv.bilateralFilter(wb, d=30, sigmaColor=80, sigmaSpace=10)
    
    # img_bilat = cv.bilateralFilter(wb, d=5, sigmaColor=20, sigmaSpace=20)
    
    bilat_end = time.time()
    
    mshift_st = time.time()
    # img_meanshift = cv.pyrMeanShiftFiltering(img_bilat, sp=45, sr=25, maxLevel=4)
    mshift_end = time.time()
    
    sobels_st = time.time()
    r_abs_grad, g_abs_grad, b_abs_grad = splitted_sobels(img_bilat)
    sobels_end = time.time()

    edges_st = time.time()
    r_edges_erode, g_edges_erode, b_edges_erode, r_edges, g_edges, b_edges = multi_edges(r_abs_grad,
                                                            g_abs_grad,
                                                            b_abs_grad)
    edges_end = time.time()        
    # Ejecutamos Hough

    hough_st = time.time()
    r_lines, g_lines, b_lines = get_rgbchannel_lines(r_edges_erode,
                                                    g_edges_erode,
                                                    b_edges_erode,
                                                    columns)
    hough_end = time.time()

    printed_lines_img = img.copy()
    
    getall_st = time.time()
    all_lines, printed_lines_img = print_lines_and_get_all_lines(r_lines, g_lines, b_lines,printed_lines_img)
    getall_end = time.time()
    
    # print(len(all_lines))
    if all_lines is not None:
        if len(all_lines) < 4:
            all_lines = []
            r_lines = []
            g_lines= []
            b_lines = []
            printed_lines_img = img.copy()
            r_lines, g_lines, b_lines = get_rgbchannel_lines2(r_edges_erode,
                                                    g_edges_erode,
                                                    b_edges_erode,
                                                    columns)
            all_lines, printed_lines_img = print_lines_and_get_all_lines(r_lines, g_lines, b_lines,printed_lines_img)

    printed_lines_img_2 = img.copy()
    
    printed_lines_img_2 = line_filtering(printed_lines_img_2, all_lines, columns)        
              
              
              
    
            
    # printed_lines_img_2 = draw_mask2(printed_lines_img_2, parallel_lines, columns)
    # Remember -> OpenCV stores things in BGR order
    lowerBound = (0, 0, 255);
    upperBound = (0, 0, 255);

    # this gives you the mask for those in the ranges you specified,
    # but you want the inverse, so we'll add bitwise_not...ç
    mask_st=time.time()
    final_mask = cv.inRange(printed_lines_img_2, lowerBound, upperBound, 1);
    erosion_type = cv.MORPH_RECT
        
        
        
        
    element = cv.getStructuringElement(erosion_type, (12,12))
    element2 = cv.getStructuringElement(erosion_type, (12,12))
    final_mask = cv.dilate(final_mask, element)  
    
    final_mask = cv.morphologyEx(final_mask, cv.MORPH_CLOSE, element2)
    img_final = img.copy()
    img_final[final_mask==255] = (0, 0, 255)
    
    mask_end=time.time()

    # res1, pl1, mean1, re1, ge1, be1 = rest_of_pipeline(img_bilat)
    # res2, final_mask, pl2, mean2, re2, ge2, be2 = rest_of_pipeline(img_bilat2)
    
    ######------- IMSHOWS--------- #######
    window_const = cv.WINDOW_AUTOSIZE
    # cv.namedWindow('resultado preprocesamiento', window_const)     
    # imshow('resultado preprocesamiento', result_prep)
    # cv.namedWindow('resultado white balancing', window_const)     
    # imshow('resultado white balancing', result_prep)
    # # # cv.namedWindow('entrada', window_const) 
    # # imshow('entrada', img)
    # # cv.namedWindow('resultado gaussian', window_const)     
    # # imshow('resultado gaussian', printed_lines_img)
    # cv.namedWindow('post-filtrado bilateral', window_const) 
    # imshow('post-filtrado bilateral', img_bilat)  
    # cv.namedWindow('post meanshift 2', window_const) 
    # imshow('post meanshift 2', img_meanshift)  
    # cv.namedWindow('post meanshift', window_const) 
    # imshow('post meanshift', img_meanshift)  
    # # imshow('grad con valor absoluto', abs_grad)
    # # # imshow('grad con valor de raiz', sqr_grad) 
    # # imshow('img_gray', img_gray)
    
    # # # imshow('res2', res2) 
    # # # imshow('resultado preprocesamiento CLAHE', result2)
    # cv.namedWindow('gradientes de red channel', window_const) 
    # imshow('gradientes de red channel', r_abs_grad)
    # cv.namedWindow('gradientes de green channel', window_const) 
    # imshow('gradientes de green channel', g_abs_grad)
    # cv.namedWindow('gradientes de blue channel', window_const) 
    # imshow('gradientes de blue channel', b_abs_grad)
    # cv.namedWindow('bordes_erode r', window_const) 
    # imshow('bordes_erode r', r_edges_erode)
    # cv.namedWindow('bordes_erode g', window_const) 
    # imshow('bordes_erode g', g_edges_erode)
    # cv.namedWindow('bordes_erode b', window_const) 
    # imshow('bordes_erode b', b_edges_erode)
    # # cv.namedWindow('lineas R', window_const) 
    # # imshow('lineas R', rp)
    # # cv.namedWindow('lineas R', window_const) 
    # # imshow('lineas G', gp)
    # # cv.namedWindow('lineas R', window_const) 
    # # imshow('lineas B', bp)
    # # cv.namedWindow('bordes r', window_const) 
    # # imshow('bordes r', r_edges)
    # # cv.namedWindow('bordes g', window_const) 
    # # imshow('bordes g', g_edges)
    # cv.namedWindow('Lineas', window_const) 
    # imshow('Lineas', printed_lines_img)
    # # imshow('Lineas ', printed_lines_img)
    cv.namedWindow('Filtro paralelas', window_const) 
    imshow('Filtro paralelas', printed_lines_img_2)

    # imshow('Imagen segmentada', segmented_img)
    # imshow('Lineas g', dst2)
    # imshow('Lineas b', dst3)
    # imshow('Lineas_erode', edges_erode)
    # im1_v = cv.vconcat([img_bilat, mean1, pl1,res1])
    # im2_v = cv.vconcat([img_bilat2, mean2, pl2, res2])
    # img_res = cv.hconcat([im1_v, im2_v])
    # img_grad1 =  cv.vconcat([re1, ge1, be1])
    # img_grad2 =  cv.vconcat([re2, ge2, be2])
    # img_gradientes =  cv.hconcat([img_grad1, img_grad2])
    cv.imwrite(save_path, img=final_mask)
    cv.imwrite(save_path2, img=img_final)
    # cv.imwrite(save_path2, img=img_gradientes)
    # cv.namedWindow('Resultado final', window_const) 
    # imshow('Resultado final', img_final)
# get the end time
    end = time.time()

    # get the execution time
    elapsed_time = end - start
    prep_time = preprocessing_end - preprocessing_st
    prep_time_per = prep_time/elapsed_time
    wb_time = wb_end - wb_st
    wb_time_per = wb_time/elapsed_time
    bilat_time = bilat_end - bilat_st
    bilat_time_per = bilat_time/elapsed_time
    mean_time = mshift_end - mshift_st
    mean_time_per = mean_time/elapsed_time
    sobel_time = sobels_end - sobels_st
    sobel_time_per = sobel_time/elapsed_time
    edge_time = edges_end - edges_st
    edge_time_per = edge_time/elapsed_time
    houg_time = hough_end - hough_st
    houg_time_per = houg_time/elapsed_time
    all_lines_time = getall_end - getall_st
    all_lines_time_per = all_lines_time/elapsed_time
    mask_time = mask_end - mask_st
    mask_time_per = mask_time/elapsed_time
    
    print('Execution time {}: '.format(n), elapsed_time, 'seconds - 100%' )
    # print('Preprocessing time ' '{}: '.format(n), prep_time, 'seconds - ', prep_time_per)
    # print('WB time ' '{}: '.format(n), wb_time, 'seconds - ', wb_time_per)
    # print('Bilat time ' '{}: '.format(n), bilat_time, 'seconds - ', bilat_time_per)
    # print('Mean time ' '{}: '.format(n), mean_time, 'seconds - ', mean_time_per)
    # print('Sobel time ' '{}: '.format(n), sobel_time, 'seconds - ', sobel_time_per)
    # print('Edge time ' '{}: '.format(n), edge_time, 'seconds - ', edge_time_per)
    # print('Hough time ' '{}: '.format(n), houg_time, 'seconds - ', houg_time_per)
    # print('All lines time ' '{}: '.format(n), all_lines_time, 'seconds - ', all_lines_time_per)
    # print('Mask time ' '{}: '.format(n), mask_time, 'seconds - ', mask_time_per)
    # print(prep_time)
    # print(wb_time)
    # print(bilat_time)
    # print(mean_time)
    # print(sobel_time)
    # print(edge_time)
    # print(houg_time)
    # print(all_lines_time)
    # print(mask_time)
    # print(prep_time_per)
    # print(wb_time_per)
    # print(bilat_time_per)
    # print(mean_time_per)
    # print(sobel_time_per)
    # print(edge_time_per)
    # print(houg_time_per)
    # print(all_lines_time_per)
    # print(mask_time_per)
 
    cv.waitKey(0)
    # # print(np.shape(l))
    cv.destroyAllWindows()


