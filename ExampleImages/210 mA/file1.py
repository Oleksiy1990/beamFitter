import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

from numpy import random
from lmfit import Parameters, minimize, fit_report, Minimizer
from scipy.ndimage.filters import gaussian_filter1d

import skimage
from skimage import io 

import os
from os.path import splitext
import sys  

def make_numpyarray(image_path_string,fullpath = False):
	""" 
	this function takes images and turns them into numpy arrays
	"""
	if fullpath:
		try:
			image_array = io.imread(image_path_string,as_grey=True)
			return image_array
		except:
			sys.exit("The image could not be read")
	else:
		path = os.getcwd() + "\\" + image_path_string
		try:
			image_array = io.imread(image_path_string,as_grey=True)
			return image_array
		except:
			sys.exit("The image could not be read")


def residual(pars, x, data=None, eps=None):
	""" 
	this function defines the Gaussian model for beam fitting and
	returns the residuals to be minimized if data are supplied, or 
	the Gaussian model itself if no data are supplied
	"""
	parvals = pars.valuesdict() # a Parameters() object is passed as "pars"
	intensity_max = parvals["I_zero"]
	centerposition = parvals["r_zero"]
	beamwidth = parvals["omega_zero"]
	bgr = parvals["backgr"]
	model = intensity_max*np.exp(-2*np.power(x-centerposition,2)/beamwidth**2) + bgr
	if data is None:
		return np.array(model)
	if eps is None:
		return np.array(model - data)
	return np.array((model - data)/eps)

def startparams_estimate(data_array,print_difference=False):
	
	# smoothened applies the Gaussian filter with width from 1/10 to 
	# 1/100 of the data array width, producing 10 arrays. This is done
	# in order to get a robust estimate without being thrown off by 
	# high frequency noise in the data
	smoothened = [gaussian_filter1d(data_array,len(data_array)/x) \
				 for x in range(10,101,10)]
	
	# we take averages of the estimates based on data with different 
	# amounts of smoothing
	peak_pos_list = np.argmax(smoothened,axis=1)
	background_list = np.amin(smoothened,axis=1)
	peak_height_list = np.amax(smoothened,axis=1) - background_list

	peak_height_estim = np.mean(peak_height_list[:5])
	peak_pos_estim = np.mean(peak_pos_list)
	background_estim = np.mean(background_list)


	
	# the estimation of the peak width only takes into account the most
	# highly smoothened data in this case
	elems_above = np.where(smoothened[9]>(peak_height_list[9]*np.exp(-0.5)+\
		background_list[9]))
	width_estim = (elems_above[0][-1]-elems_above[0][0])
	
	return (peak_height_estim,peak_pos_estim,width_estim,background_estim)


if __name__ == "__main__":

	filelist = [file for file in os.listdir(os.getcwd()) if file.endswith(".bmp") ]
	
	for file in filelist:
		pic_data = make_numpyarray(file)
		pic_data_axis = np.sum(pic_data,axis = 1)
		x_img = np.linspace(1,len(pic_data_axis),len(pic_data_axis))

		estimates = startparams_estimate(pic_data_axis)

		p_fit_img = Parameters()
		p_fit_img.add('I_zero',value=estimates[0],min=0,max=np.amax(pic_data_axis))
		p_fit_img.add('r_zero',value=estimates[1],min=1,max=len(pic_data_axis))
		p_fit_img.add('omega_zero',value=estimates[2],min=1,max=len(pic_data_axis))
		p_fit_img.add('backgr',value=estimates[3])

		smoothened_test = gaussian_filter1d(pic_data_axis,100)

		fit1 = Minimizer(residual,p_fit_img,fcn_args=(x_img,),fcn_kws={"data":pic_data_axis})
		fit1_res = fit1.minimize(method="nelder")
		fitted_func = residual(fit1_res.params,x_img)
		
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(x_img,pic_data_axis,"r-",x_img,fitted_func,"b-")
		fig.savefig("axis"+str(1)+"\\"+splitext(file)[0]+".png")
		plt.close(fig)


	#print(np.where(smoothened_test>estimates[0]*np.exp(-1)+estimates[2]))
		print(fit_report(fit1_res))
	#print(fit1_res.success)
	#print(fit1_res.message)
	

	for file in os.listdir(os.getcwd()):
		if file.endswith(".bmp"):
			print(splitext(file)[0])
	print(os.getcwd())

	sys.exit(0)
	#plt.plot(pic_data0)
	plt.subplot(211)
	plt.plot(x_img,gaussian_filter1d(pic_data_axis,100),"r-")


	plt.subplot(212)
	plt.plot(x_img,pic_data_axis,"r-",x_img,fitted_func,"b-")


	#plt.show()

	sys.exit(0)

	