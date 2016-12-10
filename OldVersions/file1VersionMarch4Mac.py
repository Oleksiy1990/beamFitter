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

def residual_2D(pars,x,y,data=None, eps=None):
	parvals = pars.valuesdict() # a Parameters() object is passed as "pars"
	intensity_max = parvals["I_zero"]
	centerposition_x = parvals["x_zero"]
	centerposition_y = parvals["y_zero"]
	beamwidth_x = parvals["omegaX_zero"]
	beamwidth_y = parvals["omegaY_zero"]
	bgr = parvals["backgr"]
	model = intensity_max*np.exp(-2*np.power(x-centerposition_x,2)/beamwidth_x**2 - \
		2*np.power(y-centerposition_y,2)/beamwidth_y**2) + bgr
	if data is None:
		return np.array(model) # we don't flatten here because this is for plotting
	if eps is None:
		resid = np.array(model - data)
		return resid.flatten() # minimization array must be flattened (LMFIT FAQ)
	else:
		resid = np.array((model - data)/eps)
		return resid.flatten()
	

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

def fit_axis(image_nparray2D,axis,minim_method="nelder"):
	axis_data = np.sum(pic_data,axis = 1) if (axis == 0) else np.sum(pic_data,axis = 0)
	axis_points = np.linspace(1,len(axis_data),len(axis_data))
	param_estimates = startparams_estimate(axis_data)
	params_for_fit = Parameters()
	params_for_fit.add('I_zero',value=param_estimates[0],min=0,max=np.amax(axis_data))
	params_for_fit.add('r_zero',value=param_estimates[1],min=1,max=len(axis_data))
	params_for_fit.add('omega_zero',value=param_estimates[2],min=1,max=len(axis_data))
	params_for_fit.add('backgr',value=param_estimates[3])
	fit = Minimizer(residual,params_for_fit,fcn_args=(axis_points,),\
		fcn_kws={"data":axis_data})
	fit_res = fit.minimize(minim_method)
	return (axis_points,axis_data,fit_res)

def format_picture(image_nparray2D,x_cent,omega_x,y_cent,omega_y,omega_fraction=2):
	length_x = image_nparray2D.shape[0]
	length_y = image_nparray2D.shape[1]
	b_left = int(x_cent - omega_fraction*omega_x) 
	b_right = int(x_cent + omega_fraction*omega_x) 
	b_bottom = int(y_cent - omega_fraction*omega_y) 
	b_top = int(y_cent + omega_fraction*omega_y) 
	b_left = b_left if (b_left >= 0) else 0
	b_right = b_right if (b_right <= length_x) else length_x
	b_bottom = b_bottom if (b_bottom >= 0) else 0
	b_top = b_top if (b_top <= length_y) else length_y
	#formatted = image_nparray2D[b_bottom:b_top,b_left:b_right]
	formatted = image_nparray2D[b_left:b_right,b_bottom:b_top]
	return formatted

def fit2D(image_nparray2D,fit_axis0=None,fit_axis1=None,minim_method="nelder"):
	if fit_axis0 is None:
		fit_axis0 = fit_axis(image_nparray2D,0,minim_method)
	if fit_axis1 is None:
		fit_axis1 = fit_axis(image_nparray2D,1,minim_method)
		
	# we first take all the initial parameters from 1D fits
	bgr2D_est = fit_axis0[2].params.valuesdict()["backgr"]/len(fit_axis1[0])
	x2D_est = fit_resultsA0[2].params.valuesdict()["r_zero"]
	omegaX2D_est = fit_axis0[2].params.valuesdict()["omega_zero"]
	y2D_est = fit_axis1[2].params.valuesdict()["r_zero"]
	omegaY2D_est = fit_axis1[2].params.valuesdict()["omega_zero"]

	smoothened_image = gaussian_filter(image_nparray2D,50)
	peakheight2D_est = np.amax(smoothened_image)
	#now we need to programatically cut out the region of interest out of the 
	#whole picture so that fitting takes way less time
		
	# NOTE! In this implementation, if the beam is small compared to picture size 
	# and is very close to the edge, the fitting will fail, because the x and y 
	# center position estimates will be off

	cropped_data = format_picture(image_nparray2D,x2D_est,omegaX2D_est,y2D_est,omegaY2D_est)
	xvals = np.linspace(1,cropped_data.shape[0],cropped_data.shape[0])
	yvals = np.linspace(1,cropped_data.shape[1],cropped_data.shape[1])
	x, y = np.meshgrid(yvals,xvals) 
	# NOTE! there's apparently some weird convention, this has to do with 
	# Cartesian vs. matrix indexing, which is explain in numpy.meshgrid manual 

	estimates_2D = Parameters()
	estimates_2D.add("I_zero",value=peakheight2D_est,min=bgr2D_est)
	estimates_2D.add("x_zero",value=0.5*len(yvals),min=0,max=len(xvals)) # NOTE! weird indexing conventions
	estimates_2D.add("y_zero",value=0.5*len(xvals),min=0,max=len(yvals)) # NOTE! weird indexing conventions
	estimates_2D.add("omegaX_zero",value=omegaX2D_est)
	estimates_2D.add("omegaY_zero",value=omegaY2D_est)
	estimates_2D.add("backgr",value=bgr2D_est)
		

	fit2D = Minimizer(residual_2D,estimates_2D,fcn_args=(x,y),fcn_kws={"data":cropped_data})
	fit_res2D = fit2D.minimize(minim_method)
	print(estimates_2D.valuesdict()["x_zero"])
	return (x,y,fit_res2D)
	


if __name__ == "__main__":
   
	imagefile = "realpic1.bmp"
	my_axis = 1
	pic_data = make_numpyarray(imagefile)

	# first do the 1D fits of the axes

	fit_resultsA0 = fit_axis(pic_data,0)
	fit_resultsA1 = fit_axis(pic_data,1)

	fit2D_result = fit2D(pic_data,fit_resultsA0,fit_resultsA1)

	plt.imshow(pic_data)
	plt.show()
	plt.close("all")

	print(fit_report(fit2D_result[2]))



	fig = plt.figure()
	ax = fig.add_subplot(111)
	Z = residual_2D(fit2D_result[2].params,fit2D_result[0],fit2D_result[1],data=None, eps=None)
	plot = ax.imshow(Z, cmap='RdBu')
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

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.pcolorfast(pic_data, cmap='RdBu')
	#plt.imshow(pic_data)
	plt.show()



	p_fit2D = Parameters()
	p_fit2D.add("I_zero",value=5)
	p_fit2D.add("x_zero",value=2)
	p_fit2D.add("y_zero",value=1)
	p_fit2D.add("omegaX_zero",value=1)
	p_fit2D.add("omegaY_zero",value=1)
	p_fit2D.add("backgr",value=1)




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

	sys.exit(0)
