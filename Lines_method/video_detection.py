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
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import ward, fcluster    


# # Create a VideoCapture object
# cap = cv.VideoCapture(0)
# Read the image
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required = True, help = "nombre de la imagen")
args = vars(ap.parse_args())
path = os.path.join('videos', args['video'])
cap = cv.VideoCapture(path)
# # Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed")
 
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
 
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv.VideoWriter('outpy.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
 
while(cap.isOpened()):
    ret, img = cap.read()
 
    if ret == True: 
     
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
#-------------------------------------  PRE-PROCESSING  ----------------------
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


        rows, columns, channel  = result_prep.shape
        features = np.zeros(shape=(rows,columns, 6))


        img_rgb = cv.bilateralFilter(result_prep, 15, 90, 90)
        img_rgb = cv.pyrMeanShiftFiltering(img_rgb, 20, 20)

    # img_rgb = result_prep
    # img_hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)


    ##We apply Sobel to each channel
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
            min_gap = int(columns*0.025)
            r_lines = cv.HoughLinesP(r_edges_erode, 1, np.pi/180, 70, 30, min_long, min_gap)
            g_lines = cv.HoughLinesP(g_edges_erode, 1, np.pi/180, 70, 30, min_long, min_gap)
            b_lines = cv.HoughLinesP(b_edges_erode, 1, np.pi/180, 70, 30, min_long, min_gap)
            
            return r_lines, g_lines, b_lines
        r_lines, g_lines, b_lines = get_rgbchannel_lines(r_edges_erode,
                                                        g_edges_erode,
                                                        b_edges_erode,
                                                        columns)


        printed_lines_img = img.copy()
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
            return all_lines, printed_lines_img    
        all_lines, printed_lines_img = print_lines_and_get_all_lines(r_lines, g_lines, b_lines,printed_lines_img)

# all_lines.append(g_lines)

# print(all_lines)

#CREAMOS VECTOR DE FEATURES
# h, s, v = cv.split(img_hsv)

# img_grad = cv.merge((h, s, v, b_abs_grad))
# img_grad = cv.merge((h, s, v, r_abs_grad, g_abs_grad, b_abs_grad))

# # print(np.shape(img_grad))
# imshow('img_grad', img_grad)

        #### -------Uso de k-means ------ #####
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

        def k_means_for_printed_lines(printed_lines_img):
            # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
            pixel_vals = printed_lines_img.reshape((-1,3))

            # Convert to float type
            pixel_vals = np.float32(pixel_vals)
            #the below line of code defines the criteria for the algorithm to stop running,
            #which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
            #becomes 85%
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)

            # then perform k-means clustering with number of clusters defined as 3
            #also random centres are initially choosed for k-means clustering
            k = 4
            retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

            # convert data into 8-bit values
            centers = np.uint8(centers)
            segmented_data = centers[labels.flatten()]

            # reshape data into the original image dimensions
            segmented_image = segmented_data.reshape((printed_lines_img.shape))

            return segmented_image

    # segmented_img = k_means_for_printed_lines(printed_lines_img)

    # # cv.imshow('imgdeg',img_grad)  
    # cv.imshow('dst',res2)  

        def findparallel(all_lines, columns):    
            # print(all_lines)
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
            return parallel_lines
    # def find_parallel_lines(lines):

            # lines_ = lines[:, 0, :]
            # angle = lines_[:, 1]

            # # Perform hierarchical clustering

            # angle_ = angle[..., np.newaxis]
            # y = pdist(angle_)
            # Z = ward(y)
            # cluster = fcluster(Z, 0.5, criterion='distance')

            # parallel_lines = []
            # for i in range(cluster.min(), cluster.max() + 1):
            #     temp = lines[np.where(cluster == i)]
            #     parallel_lines.append(temp.copy())

            # return parallel_lines

        printed_lines_img_2 = img.copy()
        # parallel_lines = find_parallel_lines(all_lines)
        parallel_lines = findparallel(all_lines, columns)
        # print(g_lines)
        # print(all_lines)
        # print(parallel_lines)
        if parallel_lines is not None:        
            
            for i in range(0, len(parallel_lines)): 
                    
                l = parallel_lines[i][0]
                # print(l[0])
                cv.line(printed_lines_img_2, (l[0], l[1]), (l[2], l[3]), (0,0,255), 2, cv.LINE_AA)




        # Remember -> OpenCV stores things in BGR order
        lowerBound = (0, 0, 255);
        upperBound = (0, 0, 255);

        # this gives you the mask for those in the ranges you specified,
        # but you want the inverse, so we'll add bitwise_not...
        final_mask = cv.inRange(printed_lines_img_2, lowerBound, upperBound, 1);
        erosion_type = cv.MORPH_RECT
        
        
        
        
        element = cv.getStructuringElement(erosion_type, (12,12))
        element2 = cv.getStructuringElement(erosion_type, (10,10))
        final_mask = cv.dilate(final_mask, element)  
        # final_mask = cv.morphologyEx(final_mask, cv.MORPH_CLOSE, element2)
        img[final_mask==255] = (0, 0, 255)  
        
            
        
        
        frame= img
        
        
        
        
        
        
        
        
        
        
        
        # Write the frame into the file 'output.avi'
        out.write(frame)
    
        # Display the resulting frame    
        # cv.imshow('frame',frame)
        print("a")
        # Press Q on keyboard to stop recording
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
 
  # Break the loop
    else:
        break 
 
# When everything done, release the video capture and video write objects
cap.release()
out.release()
 
# Closes all the frames
cv.destroyAllWindows()