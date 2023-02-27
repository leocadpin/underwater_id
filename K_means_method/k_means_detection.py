import cv2 as cv
import os
import argparse
import numpy as np
from cv2 import split
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "nombre de la imagen")
# ap.add_argument("-n", "--n_image", required=True, help= "Número de imagen")
args = vars(ap.parse_args())
path = os.path.join('images', 'img{}.png'.format(args['image']))
# n = args['n_image']
# save_path = os.path.join('results', '{}.png'.format(n))
# save_path2 = os.path.join('results', 'colour{}.png'.format(n))
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

def multi_edges(r_abs_grad, g_abs_grad, b_abs_grad):
    aperture_size = 3
    histeresys_min_thres = 150
    histeresys_max_thres = 180
    r_edges = cv.Canny(r_abs_grad, histeresys_min_thres, histeresys_max_thres, None, aperture_size)
    g_edges = cv.Canny(g_abs_grad, histeresys_min_thres, histeresys_max_thres , None, aperture_size)
    b_edges = cv.Canny(b_abs_grad, histeresys_min_thres, histeresys_max_thres, None, aperture_size)
    r_edges_erode = morphology_filters(r_edges)
    g_edges_erode = morphology_filters(g_edges)
    b_edges_erode = morphology_filters(b_edges)
    return r_edges_erode, g_edges_erode, b_edges_erode, r_edges, g_edges, b_edges


def largest_mask(masks):
    areas = [np.sum(mask) for mask in masks]
    return np.argmax(areas)

def hsv_gradient_segmentation(img, num_clusters=3):
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

    # Compute feature vector using gradient and HSV values
    feature_vector = np.concatenate((pixels, r_edges_erode.flatten().reshape((-1, 1))), axis=1)
    feature_vector = np.concatenate((feature_vector, g_edges_erode.flatten().reshape((-1, 1))), axis=1)
    feature_vector = np.concatenate((feature_vector, b_edges_erode.flatten().reshape((-1, 1))), axis=1)

    # Convert feature vector values to float32 for clustering
    feature_vector = np.float32(feature_vector)
    print(feature_vector.shape)
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

    # # Combine masks to create final segmentation mask
    segmentation_mask = np.zeros_like(labels)
    # segmentation_mask2 = np.zeros_like(labels)
    # segmentation_mask3 = np.zeros_like(labels)
    # for mask in masks:
    #     segmentation_mask = cv.bitwise_or(segmentation_mask, mask)
    # print(masks)
    biggest_area_index = largest_mask(masks)
    print(biggest_area_index)
    for i in range(num_clusters):
        if i== biggest_area_index:
            continue
        else:
            segmentation_mask = cv.bitwise_or(segmentation_mask, masks[i])
            
    segmentation_masks = [None] * num_clusters      
    for i in range(num_clusters):
        
        # segmentation_mask3 = cv.bitwise_or(segmentation_mask3, masks[2])        
        segmentation_masks[i] = np.zeros_like(labels)
        # segmentation_mask2 = np.zeros_like(labels)
        # segmentation_mask3 = np.zeros_like(labels)
        # segmentation_mask4 = np.zeros_like(labels)
        segmentation_masks[i] = cv.bitwise_or(segmentation_masks[i], masks[i])
        # segmentation_mask2 = cv.bitwise_or(segmentation_mask2, masks[1])
        # segmentation_mask3 = cv.bitwise_or(segmentation_mask3, masks[2])
        # segmentation_mask4 = cv.bitwise_or(segmentation_mask4, masks[3])
        # Convert mask to uint8 data type
        segmentation_masks[i] = segmentation_masks[i].astype(np.uint8)
        # segmentation_mask1 = segmentation_mask1.astype(np.uint8)
        # segmentation_mask2 = segmentation_mask2.astype(np.uint8)
        # segmentation_mask3 = segmentation_mask3.astype(np.uint8)
        cv.imshow('cluster {}'.format(i), segmentation_masks[i])    
    
    
    
    # segmentation_mask4 = segmentation_mask4.astype(np.uint8)
    # segmentation_mask3 = segmentation_mask3.astype(np.uint8)
    
    
    
    
    # cv.imshow('cluster4', segmentation_mask4)
    cv.waitKey(0)
    cv.destroyAllWindows()    
    
    # Apply mask to original image
    # segmented_img = cv.bitwise_and(img, img, mask=segmentation_mask1)
    img[segmentation_mask==255] = (0, 0, 255)
    # Convert segmented image back to BGR color space
    # segmented_img = cv.cvtColor(segmented_img, cv.COLOR_HSV2BGR)

    return img, segmentation_mask, segmentation_masks

def white_balance(img):
    result = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv.cvtColor(result, cv.COLOR_LAB2BGR)
    return result 
 
def cluster_classification(clusters):
    # Iterate through each cluster and perform shape validation
    result_clusters = []
    
    for cluster in clusters:
        cluster = cluster.astype(np.uint8)
        # Compute the contour of the binary image corresponding to the cluster
        contour, _ = cv.findContours(cluster, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contour = contour[0]
        
        # Create a binary image from the contour
        binary_image = np.zeros(cluster.shape, dtype=np.uint8)
        cv.drawContours(binary_image, [contour], -1, 255, -1)
        
        # Extract parallel line segments from the binary image using the Hough transform
        lines = cv.HoughLines(binary_image, rho=1, theta=np.pi/180, threshold=50)
        segments = []
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                segments.append((length, (x1, y1), (x2, y2)))
        
        # Build the angular histogram
        num_bins = 36  # Number of bins in the angular histogram
        bin_size = np.pi / (2*num_bins)  # Size of each bin
        hist = np.zeros(num_bins)
        for segment in segments:
            length = segment[0]
            x1, y1 = segment[1]
            x2, y2 = segment[2]
            angle = np.arctan2(y2-y1, x2-x1)
            if angle < 0:
                angle += np.pi
            bin_idx = int(angle / bin_size)
            hist[bin_idx] += length**2
        
        # Check if the histogram is "peaked"
        hist_mean = np.mean(hist)
        sigma_th = 0.5  # Acceptance threshold for histogram peakiness
        hist_peakiness = 0.0
        if np.max(hist) != 0:
            hist_peakiness = 1/num_bins * np.sum(hist) / (np.max(hist) + hist_mean)
        if hist_peakiness >= sigma_th:
            result_clusters.append(cluster)
    return result_clusters


if __name__== '__main__':

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
    img_res, seg, masks = hsv_gradient_segmentation(img_bilat)
    clusters_res = cluster_classification(masks)
    print(len(clusters_res))
    img_final = img[seg==255] = (0, 0, 255)
    cv.imshow('hala', img_res)
    # cv.imshow('hala2', img_fin)
    # cv.imshow('hala3', seg)
    cv.waitKey(0)
    cv.destroyAllWindows()