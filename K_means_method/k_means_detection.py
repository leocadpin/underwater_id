import cv2 as cv
import os
import argparse
import numpy as np
from cv2 import split
import time as tm
# Read the image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "nombre de la imagen")
# ap.add_argument("-n", "--n_image", required=True, help= "Número de imagen")
args = vars(ap.parse_args())
path = os.path.join('images', 'img{}.png'.format(args['image']))
n = args['image']
save_path = os.path.join('results', '{}.png'.format(n))
save_path2 = os.path.join('results', 'colour{}.png'.format(n))
img = cv.imread(path)

def preprocessing(img):
    # img2 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    img_lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    # l_blur_st = time.time()
    rows, columns, channel  = img_lab.shape
    
    l, a, b = cv.split(img_lab)

    # l_blur_st = time.time()
    l_blur = cv.medianBlur(l, 5)
    # l_blur_end = time.time()
    # cielab_st = time.time()
    # for row in range(rows):
        # for column in range(columns):                    
    img_lab[:, :, 0] = (1.5*img_lab[:, :, 0]) - (0.5*l_blur[:, :]) 
            
    # result = cv.cvtColor(img_lab, cv.COLOR_LAB2RGB)
    # result_gr = cv.cvtColor(img_lab, cv.COLOR_RGB2GRAY)
    
    # cielab_end = time.time()
    # create a CLAHE object (Arguments are optional).
    # clahe_st = time.time()
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    img_lab[...,0] = clahe.apply(img_lab[...,0])
    # clahe_end = time.time()

    # blur_time = l_blur_end - l_blur_st
    # cielab_time = cielab_end - cielab_st
    # clahe_time = clahe_end - clahe_st
    # print('BLUR time ' '{}: '.format(n), blur_time, 'seconds - ', _time_per)
    # print('CIELAB time ' '{}: '.format(n), cielab_time, 'seconds - ', _time_per)
    # print('CLAHE time ' '{}: '.format(n), clahe_time, 'seconds - ', _time_per)
    result_prep = cv.cvtColor(img_lab, cv.COLOR_LAB2RGB)
    return result_prep

def splitted_sobels(img_rgb):
    # Conversion a escala de grises 
    r_img , g_img, b_img = split(img_rgb)


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

def morphology_filters(edges):
    # Forma del filtro
    erosion_type = cv.MORPH_RECT
    erosion_type2 = cv.MORPH_ELLIPSE
    # El último parámetro es el tamaño del filtro, en este caso 5x5
    # element = cv.getStructuringElement(erosion_type, (4,4)) 
    # element2 = cv.getStructuringElement(erosion_type2, (4,4)) 
    element3 = cv.getStructuringElement(erosion_type2, (2,2))
    element4 = cv.getStructuringElement(erosion_type, (1,1)) 
    # dst = cv.erode(dst,element2)
    # dst = cv.morphologyEx(img_meanshift, cv.MORPH_CLOSE, element2)
    # dst2 = cv.morphologyEx(dst, cv.MORPH_CLOSE, element2)
    edges_erode = cv.dilate(edges, element4)
    edges_erode = cv.dilate(edges_erode, element4)

    edges_erode = cv.erode(edges_erode, element3)

    return edges_erode

def morphology_filters3(edges):
    # Forma del filtro
    erosion_type = cv.MORPH_RECT
    erosion_type2 = cv.MORPH_ELLIPSE
 
    element = cv.getStructuringElement(erosion_type2, (15,15))
    element2 = cv.getStructuringElement(erosion_type, (1,1)) 
    element3 = cv.getStructuringElement(erosion_type, (3,3)) 
    edges_erode = cv.erode(edges.astype(np.uint8), element2)
    edges_erode = cv.erode(edges_erode, element3)
    edges_erode = cv.erode(edges_erode, element3)
    edges_erode = cv.erode(edges_erode, element3)
    edges_erode = cv.erode(edges_erode, element2)
    edges_erode = cv.erode(edges_erode, element2)
    edges_erode = cv.erode(edges_erode, element2)
    edges_erode = cv.erode(edges_erode, element2)
    edges_erode = cv.dilate(edges_erode, element)
    

    return edges_erode

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
    
    # cv.imshow('Gabor_r', gabor_r)
    # cv.imshow('Gabor_g', gabor_g)
    # cv.imshow('Gabor_b', gabor_b)
    # cv.imshow('Canny r', r_edges)
    # cv.imshow('Canny g', g_edges)
    # cv.imshow('Canny b', b_edges)
    # cv.imshow('cleaner Canny r', r_edges_erode)
    # cv.imshow('cleaner g', g_edges_erode)
    # cv.imshow('cleaner b', b_edges_erode)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return r_edges_erode, g_edges_erode, b_edges_erode, r_edges, g_edges, b_edges
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

# def largest_mask(masks):
#     areas = [np.sum(mask) for mask in masks]
#     return np.argmax(areas)

# def smallest_mask(masks):
#     areas = [np.sum(mask) for mask in masks]
#     return np.argmin(areas)

def second_smallest_mask(masks):
    areas = [np.sum(mask) for mask in masks]
    smallest_idx = np.argmin(areas)
    areas[smallest_idx] = np.inf
    second_smallest_idx = np.argmin(areas)
    return second_smallest_idx
def get_largest_contour(mask):
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if(len(contours)<=2 and not None):
        return mask
    if not contours:
        return None
    max_contour = max(contours, key=cv.contourArea)
    contour_mask = cv.drawContours(np.zeros_like(mask), [max_contour], 0, 255, -1)
    return contour_mask


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
        # print("area de la máscara ",i,": ", area_percent)
        if min_area_percent <= area_percent <= max_area_percent:
            filtered_masks.append(mask)
        i=i+1 
    return filtered_masks

def count_lines(mask):
    edges = cv.Canny(mask.astype(np.uint8), 100, 200)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 10, minLineLength=60, maxLineGap=10)
    if lines is not None:
        mask_lines = mask.copy()
        mask_lines = mask_lines.astype(np.uint8)
        mask_lines = cv.cvtColor(mask_lines, cv.COLOR_GRAY2BGR)
        # all_lines = b_lines
        # for i in range(0, len(lines)):
        #     l = lines[i][0]
        #     cv.line(mask_lines, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv.LINE_AA)
        
        # cv.imshow('mask lines', mask_lines)    
    
        # cv.waitKey(0)
        # cv.destroyAllWindows()    
            
        return len(lines)

def filter_masks_by_lines(masks, min_lines, max_lines):
    filtered_masks = []
    for mask in masks:
        num_lines = count_lines(mask)
        if num_lines is not None:
            if num_lines >= min_lines and num_lines <= max_lines:
                filtered_masks.append(mask)
                
    return filtered_masks

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
    # lab_img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    # hls_img = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    # xyz_img = cv.cvtColor(img, cv.COLOR_BGR2XYZ)
    # rgb_img = img.copy()
    # Reshape the image to a 2D array of pixels
    h,w, _ = hsv_img.shape
    area_img = h*w
    pixels = hsv_img.reshape((-1, 3))
    # pixels_lab = lab_img.reshape((-1, 3))
    # pixels_hls = hls_img.reshape((-1, 3))
    # pixels_rgb = rgb_img.reshape((-1, 3))
    # pixels_xyz = xyz_img.reshape((-1, 3))
    # Compute feature vector using gradient and HSV values
    # feature_vector = np.concatenate((pixels, pixels_xyz), axis=1)
    feature_vector = np.concatenate((pixels, r_edges_erode.flatten().reshape((-1, 1))), axis=1)
    feature_vector = np.concatenate((feature_vector, g_edges_erode.flatten().reshape((-1, 1))), axis=1)
    feature_vector = np.concatenate((feature_vector, b_edges_erode.flatten().reshape((-1, 1))), axis=1)

    # Convert feature vector values to float32 for clustering
    feature_vector = np.float32(feature_vector)
    # print(feature_vector.shape)
    # Define criteria and apply k-means clustering
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1)
    k_mean_start = tm.time()
    ret, labels, centers = cv.kmeans(feature_vector, num_clusters, None, criteria, 10, cv.KMEANS_PP_CENTERS)
    k_mean_end = tm.time()
    k_time = k_mean_end - k_mean_start
    # Reshape the labels to match the original image shape
    labels = labels.reshape(img.shape[:2])
    segmentation_mask = np.zeros_like(labels)
    # Create a mask for each cluster
    masks = []
    
    for i in range(num_clusters):
        mask = np.zeros_like(labels)
        mask[labels == i] = 255
        masks.append(mask)
        
    filtered_masks = filter_masks_by_area(masks,area_img, 4, 25)
    # print(len(filtered_masks))
    if len(filtered_masks)>=1:
        filtered_by_lines = filter_masks_by_lines(filtered_masks, 1, 10)
        # print(len(filtered_by_lines))
        if len(filtered_by_lines) >= 1:
            for i in range(len(filtered_by_lines)):
                segmentation_mask = cv.bitwise_or(segmentation_mask, filtered_by_lines[i])
        else:
            for i in range(len(filtered_masks)):
                segmentation_mask = cv.bitwise_or(segmentation_mask, filtered_masks[i])        

    

    # segmentation_masks = [None] * num_clusters      
    # for i in range(num_clusters):           
    #     segmentation_masks[i] = np.zeros_like(labels)         
    #     segmentation_masks[i] = cv.bitwise_or(segmentation_masks[i], masks[i])            
    #     segmentation_masks[i] = segmentation_masks[i].astype(np.uint8)   
    #     cv.imshow('cluster {}'.format(i), segmentation_masks[i])    
    
    # hache, ese, uve = split(hsv_img)
    # cv.imshow('hsv', hsv_img)
    # cv.imshow('h', hache)
    # cv.imshow('s', ese)
    # cv.imshow('v', uve)
    # segmentation_mask4 = segmentation_mask4.astype(np.uint8)
    # segmentation_mask3 = segmentation_mask3.astype(np.uint8)
    
    
    
    
    # # cv.imshow('cluster4', segmentation_mask4)
    # cv.waitKey(0)
    # cv.destroyAllWindows()    
    
    # Apply mask to original image
    # segmented_img = cv.bitwise_and(img, img, mask=segmentation_mask1)
    # img[segmentation_mask==255] = (0, 0, 255)
    # Convert segmented image back to BGR color space
    # segmented_img = cv.cvtColor(segmented_img, cv.COLOR_HSV2BGR)

    return segmentation_mask, k_time

def white_balance(img):
    result = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv.cvtColor(result, cv.COLOR_LAB2BGR)
    return result 
 
def morphology_filters(img_bin):
        # Forma del filtro
    erosion_type = cv.MORPH_RECT
    erosion_type2 = cv.MORPH_ELLIPSE
    # El último parámetro es el tamaño del filtro, en este caso 5x5
    # element = cv.getStructuringElement(erosion_type, (4,4)) 
    # element2 = cv.getStructuringElement(erosion_type2, (4,4)) 
    element3 = cv.getStructuringElement(erosion_type2, (5,5))
    element4 = cv.getStructuringElement(erosion_type, (2,2)) 
    # dst = cv.erode(dst,element2)
    # dst = cv.morphologyEx(img_meanshift, cv.MORPH_CLOSE, element2)
    # dst2 = cv.morphologyEx(dst, cv.MORPH_CLOSE, element2)
    edges_erode = cv.dilate(img_bin.astype(np.uint8), element4)
    
    edges_erode = cv.dilate(edges_erode, element4)
    edges_erode = cv.erode(edges_erode, element3)
    
   
    return edges_erode

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
 
def get_best_rectangular_contour(mask1, mask2):
    """
    Compara dos máscaras binarias y devuelve la que tiene el mejor contorno rectangular.
    
    Args:
        mask1 (ndarray): La primera máscara binaria.
        mask2 (ndarray): La segunda máscara binaria.
        
    Returns:
        ndarray: La máscara binaria que tiene el mejor contorno rectangular.
    """
    
    # Encuentra los contornos de cada máscara
    contours1, _ = cv.findContours(mask1.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv.findContours(mask2.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Encuentra el rectángulo delimitador mínimo para cada contorno
    rects1 = [cv.boundingRect(contour) for contour in contours1]
    rects2 = [cv.boundingRect(contour) for contour in contours2]
    
    # Calcula el área del rectángulo delimitador mínimo para cada contorno
    areas1 = [rect[2] * rect[3] for rect in rects1]
    areas2 = [rect[2] * rect[3] for rect in rects2]
    
    # Encuentra el índice del contorno con el área más grande para cada máscara
    max_index1 = np.argmax(areas1)
    max_index2 = np.argmax(areas2)
    
    # Compara los rectángulos delimitadores mínimos de cada contorno y devuelve la máscara
    # que tiene el rectángulo más grande
    if areas1[max_index1] > areas2[max_index2]:
        return mask1
    else:
        return mask2



if __name__== '__main__':

    start = tm.time()
    img_prep = preprocessing(img)
    img_wb = white_balance(img_prep)
        #  = cv.GaussianBlur(wb,(3,3),cv.BORDER_WRAP)
    # Convert to grayscale
    gray = cv.cvtColor(img_wb, cv.COLOR_BGR2GRAY)

    # Apply median blur to reduce noise
    blur = cv.medianBlur(gray, 5)
    blur=  cv.cvtColor(blur, cv.COLOR_GRAY2BGR)
    img_bilat = cv.bilateralFilter(img_wb, d=9, sigmaColor=75, sigmaSpace=75)
    
    # img_meanshift = cv.pyrMeanShiftFiltering(img_bilat, sp=45, sr=25, maxLevel=4)
    final_mask, k_means_execution_time  = hsv_gradient_segmentation(img_bilat)
    # clusters_res = cluster_classification(masks)
    # print(len(clusters_res))
    final_mask = morphology_filters3(final_mask)
    final_mask = get_largest_contour(final_mask)
    img_final = img.copy()
    img_final[final_mask==255] = (0, 0, 255)
    end = tm.time()
    
    execution_time = end - start
    print('Tiempo de ejecución: ', execution_time)
    print('Tiempo de ejecución K-means: ', k_means_execution_time)
    cv.imwrite(save_path, img=final_mask)
    cv.imwrite(save_path2, img=img_final)
    # cv.imshow('hala', img_final)
    # cv.imshow('hala2', img_fin)
    # cv.imshow('hala3', seg)
    cv.waitKey(0)
    cv.destroyAllWindows()