import cv2 as cv
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description = 'Programa para obtener el IoU de dos imágenes')
parser.add_argument("-n", "--n_image", required=True, help= "Número de imagen")

args = vars(parser.parse_args())

n = args['n_image']
obtenida_path = os.path.join( 'results','{}.png'.format(n))
ideal_path = os.path.join('ideal_output', '{}.png'.format(n))

obtenida = cv.imread(obtenida_path, cv.IMREAD_GRAYSCALE)
solucion = cv.imread(ideal_path, cv.IMREAD_GRAYSCALE)

# hala=cv.addWeighted(obtenida, 0.5, solucion, 0.5, gamma=1)
# window_const = cv.WINDOW_AUTOSIZE
# cv.namedWindow('hala', window_const)  
# cv.imshow('hala',hala)
# cv.waitKey(0)
# cv.destroyAllWindows()
if (obtenida is None) or (solucion is None):
        print ('Error en alguna de las imagenes')
        quit()
  
# Calculamos intersection over union (IoU), es el porcentaje de acierto de nuestro programa 
intersectionAB = cv.countNonZero(obtenida & solucion)
unionAB = cv.countNonZero(obtenida | solucion)
IoU = intersectionAB / unionAB
    
print(IoU)