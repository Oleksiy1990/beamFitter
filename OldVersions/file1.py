import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from numpy import random
from lmfit import Parameters, minimize, fit_report, Minimizer
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter

import skimage
from skimage import io 

import os
import sys  

from mathmodels import residual_G1D, residual_G2D
import fitterroutines as frt

def make_numpyarray(image_path_string,fullpath = True):
	""" 
	this function takes a string specifying image path 
	and turns images into numpy arrays

	make_numpyarray(image_path_string,fullpath = True)
	"""
	if fullpath:
		try:
			image_array = io.imread(image_path_string,as_grey=True)
			return image_array
		except:
			print(fullpath)
			sys.exit("The image could not be read")
	else:
		path = os.getcwd() + "/" + image_path_string
		try:
			image_array = io.imread(image_path_string,as_grey=True)
			return image_array
		except:
			print(path)
			sys.exit("The image could not be read")


	
if __name__ == "__main__":


	imagefile = "realpic1.bmp"
	my_axis = 1
	pic_data = make_numpyarray(imagefile,fullpath=False)

	# first do the 1D fits of the axes

	fit_resultsA0 = frt.fit_axis(pic_data,0)
	fit_resultsA1 = frt.fit_axis(pic_data,1)

	fit2D_result = frt.fit2D(pic_data,fit_resultsA0,fit_resultsA1,omega_fraction=2)

	plt.imshow(pic_data)
	plt.show()
	plt.close("all")

	print(fit_report(fit2D_result[2]))



	fig = plt.figure()
	ax = fig.add_subplot(111)
	Z = residual_G2D(fit2D_result[2].params,fit2D_result[0],fit2D_result[1],data=None, eps=None)
	plot = ax.pcolorfast(Z, cmap='RdBu')
	plt.colorbar(plot)
	plt.show()
	plt.close("all")

	sys.exit(0)

	#formatted_picture = format_picture(pic_data,x2D,omegaX,y2D,omegaY,omega_fraction=2)
	#print(pic_data.shape)
	#print(formatted_picture.shape)
	#print(pic_data.shape[0] == formatted_picture.shape[0])


	#fig = plt.figure()
	#ax = fig.add_subplot(111)
	#plot = ax.pcolorfast(formatted_picture, cmap='rainbow')
	#plt.colorbar(plot)
	#plt.show()
	#plt.close("all")



	#plt.plot(fit_results[0],fit_results[1],"r-",fit_results[0],residual(fit_results[2].params,fit_results[0]),"b-")
	#plt.show()

	
	
	#sys.exit(0)




	smoothened_image = gaussian_filter(formatted_picture,50)
	xvals = np.linspace(1,formatted_picture.shape[0],formatted_picture.shape[0])
	yvals = np.linspace(1,formatted_picture.shape[1],formatted_picture.shape[1])
	x, y = np.meshgrid(yvals,xvals)

	print(x.shape)
	print(y.shape)
	print(formatted_picture.shape)
	#sys.exit(0)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	Z = residual_2D(fit_res2D.params,x,y,data=None, eps=None)
	plot = ax.pcolorfast(Z, cmap='RdBu')
	plt.colorbar(plot)
	plt.show()
	plt.close("all")
	

	sys.exit(0)


	p_fit2D = Parameters()
	p_fit2D.add("I_zero",value=5)
	p_fit2D.add("x_zero",value=5)
	p_fit2D.add("y_zero",value=1)
	p_fit2D.add("omegaX_zero",value=3)
	p_fit2D.add("omegaY_zero",value=1)
	p_fit2D.add("backgr",value=1)
	p_fit2D.add("theta_rot",value=0.2*np.pi/3)

	x,y = np.meshgrid(np.linspace(0,10,100),np.linspace(0,4,40))
	fig = plt.figure()
	ax = fig.add_subplot(111)
	Z = residual_G2D(p_fit2D,x,y,data=None, eps=None)
	plot = ax.pcolorfast(Z, cmap='RdBu')
	plt.colorbar(plot)
	plt.show()
	plt.close("all")
	
	sys.exit(0)

	
	#this is the way to get numerical values of fit parameters
	print(fit1_res.params.valuesdict()["omega_zero"])
	for file in os.listdir(os.getcwd()):
		if file.endswith(".jpg"):
			print(file)
	print(os.getcwd())

	
	#plt.plot(pic_data0)
	plt.subplot(211)
	plt.plot(axes_pts[0],gaussian_filter1d(pic_data_both_axes[0],100),"r-")


	plt.subplot(212)
	plt.plot(axes_pts[0],pic_data_both_axes[0],"r-",axes_pts[0],fitted_func,"b-")


	plt.show()
	# some comments for GitHub test

	sys.exit(0)
