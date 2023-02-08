#!/usr/bin/env python3
from ast import Pass
from logging import info
from typing_extensions import Self
from unittest import skip
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Empty
from cv_bridge import CvBridge
import cv2 
import os
import numpy as np
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Point, Twist




class Nodo(object):
    
    def __init__(self):
        # Params
        self.r = rospy.Rate(1)
        self.image = None
        self.br = CvBridge()
        self.cmd = Twist()
        
        
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(1)
        self.image_rgb = None
        
        # Publishers
       
        # TODO Saber en que topic se publica 
        # self.pub = rospy.Publisher('', , queue_size=5)
        
        

        # TODO Saber cual es el topic que publica la imagen de la camara
        self.sub = rospy.Subscriber(
            "/camera/depth/image_raw", Image, self.callback)
        # self.sub2 = rospy.Subscriber("/camera/rgb/image_raw",Image,self.callback2)
        
        
        

    
    #Leemos imagen de la camara uOne
    def callback(self, msg):
        # rospy.loginfo('Image received...')
        self.image = self.br.imgmsg_to_cv2(msg)
 
    
    def start(self):
            
            
        while True:
            
            if self.image is not None:
                
                #TODO: AQUI VA EL ALGORITMO DE VISIÓN
                a=0
                
                
                    
                               
                      
            
                                                                    

    


if __name__ == '__main__':

    rospy.init_node("pipe_detector", anonymous=True)
    my_node = Nodo()
    my_node.start()
    