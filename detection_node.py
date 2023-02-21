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
        # self.sub = rospy.Subscriber(
        #     "/uone/stereo/left/image", Image, self.callback)
        self.sub = rospy.Subscriber(
            "bslmaris/central_image_raw", Image, self.callback)
        
        # self.sub2 = rospy.Subscriber("/camera/rgb/image_raw",Image,self.callback2)
        
        #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
        

    
    #Leemos imagen de la camara uOne
    def callback(self, msg):
        rospy.loginfo('Image received...')
        self.image = self.br.imgmsg_to_cv2(msg, 'rgb8')

    
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
                
                # img_rgb = result_prep
                # img_hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)

            #--------------Clusterización -----------------------------------------#


                rows, columns, channel  = result_prep.shape
                features = np.zeros(shape=(rows,columns, 6))


                img_rgb = cv.bilateralFilter(result_prep, 15, 90, 90)
                img_rgb = cv.pyrMeanShiftFiltering(img_rgb, 20, 20)

            ##We apply Sobel to each channel

                r_abs_grad, g_abs_grad, b_abs_grad = splitted_sobels(img_rgb)
                


                r_edges_erode, g_edges_erode, b_edges_erode, r_edges, g_edges, b_edges = multi_edges(r_abs_grad,
                                                                        g_abs_grad,
                                                                        b_abs_grad)
                
                # Ejecutamos Hough


                r_lines, g_lines, b_lines = get_rgbchannel_lines(r_edges_erode,
                                                                g_edges_erode,
                                                                b_edges_erode,
                                                                columns)


                printed_lines_img = img.copy()

                all_lines, printed_lines_img = print_lines_and_get_all_lines(r_lines, g_lines, b_lines,printed_lines_img)
                
            
            
        #         def k_means_for_feature_vector(img_grad):
        #             # Z = img_grad.reshape((-1,4))
        #             Z = img_grad.reshape((-1,6))

        #             # # convert to np.float32
        #             Z = np.float32(Z)
        #             # # define criteria, number of clusters(K) and apply kmeans()
        #             criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        #             K = 4
        #             ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
        #             # # Now convert back into uint8, and make original image
        #             center = np.uint8(center)
        #             res = center[label.flatten()]
        #             res2 = res.reshape((img_grad.shape))

        #             # r,g,b,grad1 = cv.split(res2)
        #             r,g,b,grad1, grad2, grad3 = cv.split(res2)

        #             res2 = merge((r,g,b))
        #             return res2

        #         def k_means_for_printed_lines(printed_lines_img):
        #             # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
        #             pixel_vals = printed_lines_img.reshape((-1,3))

        #             # Convert to float type
        #             pixel_vals = np.float32(pixel_vals)
        #             #the below line of code defines the criteria for the algorithm to stop running,
        #             #which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
        #             #becomes 85%
        #             criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)

        #             # then perform k-means clustering with number of clusters defined as 3
        #             #also random centres are initially choosed for k-means clustering
        #             k = 4
        #             retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

        #             # convert data into 8-bit values
        #             centers = np.uint8(centers)
        #             segmented_data = centers[labels.flatten()]

        #             # reshape data into the original image dimensions
        #             segmented_image = segmented_data.reshape((printed_lines_img.shape))

        #             return segmented_image
            

                
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