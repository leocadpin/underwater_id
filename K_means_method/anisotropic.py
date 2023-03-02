import numpy as np
import cv2 as cv
from cv2 import split
import os
import argparse
import scipy.ndimage.filters as flt
import warnings
from medpy.filter.smoothing import anisotropic_diffusion
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "nombre de la imagen")
args = vars(ap.parse_args())
path = os.path.join('images', 'img{}.png'.format(args['image']))
n = args['image']
save_path = os.path.join('results', '{}.png'.format(n))
save_path2 = os.path.join('results', 'colour{}.png'.format(n))
img = cv.imread(path)

def white_balance(img):
    result = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv.cvtColor(result, cv.COLOR_LAB2BGR)
    return result 
def anisotropic_filter(img, iterations=1, kappa=50, gamma=0.1, option=1):
    # Create 3x3 horizontal and vertical sobel filters
    hx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    hy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    rows, cols, _ = img.shape
    img_filtered = np.copy(img)

    for i in range(iterations):
        # Compute gradient magnitude and orientation
        grad_x = cv.filter2D(img_filtered, -1, hx)
        grad_y = cv.filter2D(img_filtered, -1, hy)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_ori = np.arctan2(grad_y, grad_x)

        # Compute diffusion coefficients
        if option == 1:
            c = 1 / (1 + (grad_mag / kappa) ** 2)
        elif option == 2:
            c = np.exp(-(grad_mag / kappa) ** 2)
        else:
            c = 1 / (1 + (grad_mag / kappa) ** 4)

        # Compute update factor
        delta = gamma * c

        # Update image
        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                diff_x = np.sin(grad_ori[x, y])
                diff_y = np.cos(grad_ori[x, y])
                diff_xy = np.sin(45 * np.pi / 180) * diff_x + np.cos(45 * np.pi / 180) * diff_y

                if np.greater(np.abs(diff_x), np.abs(diff_xy)).any():
                    diff = diff_x * delta[x, y]
                else:
                    diff = diff_xy * delta[x, y]

                img_filtered[x, y] += diff

    return img_filtered

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

def multi_edges(r_abs_grad, g_abs_grad, b_abs_grad):
    aperture_size = 3
    histeresys_min_thres = 60
    histeresys_max_thres = 100
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
def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),sigma=0, option=1,ploton=False):
	"""
	Anisotropic diffusion.

	Usage:
	imgout = anisodiff(im, niter, kappa, gamma, option)

	Arguments:
	        img    - input image
	        niter  - number of iterations
	        kappa  - conduction coefficient 20-100 ?
	        gamma  - max value of .25 for stability
	        step   - tuple, the distance between adjacent pixels in (y,x)
	        option - 1 Perona Malik diffusion equation No 1
	                 2 Perona Malik diffusion equation No 2
	        ploton - if True, the image will be plotted on every iteration

	Returns:
	        imgout   - diffused image.

	kappa controls conduction as a function of gradient.  If kappa is low
	small intensity gradients are able to block conduction and hence diffusion
	across step edges.  A large value reduces the influence of intensity
	gradients on conduction.

	gamma controls speed of diffusion (you usually want it at a maximum of
	0.25)

	step is used to scale the gradients in case the spacing between adjacent
	pixels differs in the x and y axes

	Diffusion equation 1 favours high contrast edges over low contrast ones.
	Diffusion equation 2 favours wide regions over smaller ones.

	Reference: 
	P. Perona and J. Malik. 
	Scale-space and edge detection using ansotropic diffusion.
	IEEE Transactions on Pattern Analysis and Machine Intelligence, 
	12(7):629-639, July 1990.

	Original MATLAB code by Peter Kovesi  
	School of Computer Science & Software Engineering
	The University of Western Australia
	pk @ csse uwa edu au
	<http://www.csse.uwa.edu.au>

	Translated to Python and optimised by Alistair Muldal
	Department of Pharmacology
	University of Oxford
	<alistair.muldal@pharm.ox.ac.uk>

	June 2000  original version.       
	March 2002 corrected diffusion eqn No 2.
	July 2012 translated to Python
	"""

	# ...you could always diffuse each color channel independently if you
	# really want
	if img.ndim == 3:
		warnings.warn("Only grayscale images allowed, converting to 2D matrix")
		img = img.mean(2)

	# initialize output array
	img = img.astype('float32')
	imgout = img.copy()

	# initialize some internal variables
	deltaS = np.zeros_like(imgout)
	deltaE = deltaS.copy()
	NS = deltaS.copy()
	EW = deltaS.copy()
	gS = np.ones_like(imgout)
	gE = gS.copy()

	# create the plot figure, if requested
	if ploton:
		import pylab as pl
		from time import sleep

		fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
		ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)

		ax1.imshow(img,interpolation='nearest')
		ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
		ax1.set_title("Original image")
		ax2.set_title("Iteration 0")

		fig.canvas.draw()

	for ii in np.arange(1,niter):

		# calculate the diffs
		deltaS[:-1,: ] = np.diff(imgout,axis=0)
		deltaE[: ,:-1] = np.diff(imgout,axis=1)

		if 0<sigma:
			deltaSf=flt.gaussian_filter(deltaS,sigma);
			deltaEf=flt.gaussian_filter(deltaE,sigma);
		else: 
			deltaSf=deltaS;
			deltaEf=deltaE;
			
		# conduction gradients (only need to compute one per dim!)
		if option == 1:
			gS = np.exp(-(deltaSf/kappa)**2.)/step[0]
			gE = np.exp(-(deltaEf/kappa)**2.)/step[1]
		elif option == 2:
			gS = 1./(1.+(deltaSf/kappa)**2.)/step[0]
			gE = 1./(1.+(deltaEf/kappa)**2.)/step[1]

		# update matrices
		E = gE*deltaE
		S = gS*deltaS

		# subtract a copy that has been shifted 'North/West' by one
		# pixel. don't as questions. just do it. trust me.
		NS[:] = S
		EW[:] = E
		NS[1:,:] -= S[:-1,:]
		EW[:,1:] -= E[:,:-1]

		# update the image
		imgout += gamma*(NS+EW)

		if ploton:
			iterstring = "Iteration %i" %(ii+1)
			ih.set_data(imgout)
			ax2.set_title(iterstring)
			fig.canvas.draw()
			# sleep(0.01)

	return imgout
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
    result_prep = white_balance(result_prep)
    return result_prep

img_prep = preprocessing(img)
img_gray = cv.cvtColor(img_prep, cv.COLOR_RGB2GRAY)
img_bilat = cv.bilateralFilter(img_prep, d=9, sigmaColor=75, sigmaSpace=75)
img_anis = cv.ximgproc.anisotropicDiffusion(img_bilat, .1, .1, 25)
# img_anis = cv.cvtColor(img_anis, cv.COLOR_GRAY2RGB)
rs, gs, bs = splitted_sobels(img_anis)
_, _, _, re, ge, be = multi_edges(rs, gs, bs)
cv.imshow('Entrada', img)
cv.imshow('prep', img_prep)
cv.imshow('Bilateral', img_bilat)
cv.imshow('Ansisotropic', img_anis)
cv.imshow('redges', re)
cv.imshow('gedges', ge)
cv.imshow('bedges', be)

# cv2.imshow('Gabor filter', filter)
cv.waitKey(0)
cv.destroyAllWindows()
