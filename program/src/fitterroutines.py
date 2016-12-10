import numpy as np

from lmfit import Parameters, minimize, fit_report, Minimizer
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from skimage import io 
import sys, os

from src.mathmodels import residual_G1D, residual_G2D, residual_G2D_norotation

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
		path = os.getcwd() + "\\" + image_path_string
		try:
			image_array = io.imread(image_path_string,as_grey=True)
			return image_array
		except:
			print(path)
			sys.exit("The image could not be read")


def startparams_estimate(data_array,print_difference=False):

	"""
	This function takes a 1D data array which is supposed to contain a 
	Gaussian peak and returns the estimates of peak height, peak position,
	peak width, and background signal. This is used as a starting point for 
	1D fitting of Gaussians

	startparams_estimate(data_array,print_difference=False)
	"""
	
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
	"""
	This function fits one axis of a 2D array representing an image by doing 
	a summation along the other axis 

	fit_axis(image_nparray2D,axis,minim_method="nelder")
	"""


	axis_data = np.sum(image_nparray2D,axis = 1) if (axis == 0) else np.sum(image_nparray2D,axis = 0)
	axis_points = np.linspace(1,len(axis_data),len(axis_data))
	param_estimates = startparams_estimate(axis_data)
	params_for_fit = Parameters()
	params_for_fit.add('I_zero',value=param_estimates[0],min=0,max=np.amax(axis_data))
	params_for_fit.add('r_zero',value=param_estimates[1],min=1,max=len(axis_data))
	params_for_fit.add('omega_zero',value=param_estimates[2],min=1,max=len(axis_data))
	params_for_fit.add('backgr',value=param_estimates[3])
	fit = Minimizer(residual_G1D,params_for_fit,fcn_args=(axis_points,),\
		fcn_kws={"data":axis_data})
	fit_res = fit.minimize(minim_method)
	return (axis_points,axis_data,fit_res)


def format_picture(image_nparray2D,x_cent,omega_x,y_cent,omega_y,omega_fraction=2):

	"""
	This function takes a 2D array and the values for center position of a Gaussian 
	peak and its width to crop out a region within a given number of omega from the 
	center (the default is 2*omega). This is useful for 2D fitting and plotting, when 
	the beam is much smaller than the entire picture size, it's much faster to 
	only fit where the peak as, without evaluating too many background points 

	format_picture(image_nparray2D,x_cent,omega_x,y_cent,omega_y,omega_fraction=2)
	"""
	
	# one has to be careful here because index 0 is not necessarily "x" in plots
	# index 0 is always rows, and that's what usually plotted vertically 
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
	formatted = image_nparray2D[b_left:b_right,b_bottom:b_top]
	return formatted



def fit2D(image_nparray2D,fit_axis0=None,fit_axis1=None,minim_method="nelder",omega_fraction=2,rotation=True):
	if fit_axis0 is None:
		fit_axis0 = fit_axis(image_nparray2D,0,minim_method)
	if fit_axis1 is None:
		fit_axis1 = fit_axis(image_nparray2D,1,minim_method)
		
	# we first take all the initial parameters from 1D fits
	bgr2D_est = fit_axis0[2].params.valuesdict()["backgr"]/len(fit_axis1[0])
	x2D_est = fit_axis0[2].params.valuesdict()["r_zero"]
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

	cropped_data = format_picture(image_nparray2D,x2D_est,omegaX2D_est,y2D_est,omegaY2D_est,omega_fraction)
	xvals = np.linspace(1,cropped_data.shape[0],cropped_data.shape[0])
	yvals = np.linspace(1,cropped_data.shape[1],cropped_data.shape[1])
	x, y = np.meshgrid(yvals,xvals) 
	# NOTE! there's apparently some weird convention, this has to do with 
	# Cartesian vs. matrix indexing, which is explain in numpy.meshgrid manual 

	estimates_2D = Parameters()
	estimates_2D.add("I_zero",value=peakheight2D_est,min=bgr2D_est)
	estimates_2D.add("x_zero",value=0.5*len(yvals),min=0,max=len(yvals)) # NOTE! weird indexing conventions
	estimates_2D.add("y_zero",value=0.5*len(xvals),min=0,max=len(xvals)) # NOTE! weird indexing conventions
	estimates_2D.add("omegaX_zero",value=omegaX2D_est)
	estimates_2D.add("omegaY_zero",value=omegaY2D_est)
	estimates_2D.add("theta_rot",value=0*np.pi,min = 0,max = np.pi) #just starting with 0
	estimates_2D.add("backgr",value=bgr2D_est)
	

	if rotation:
		fit2D = Minimizer(residual_G2D,estimates_2D,fcn_args=(x,y),fcn_kws={"data":cropped_data})
		print("Including rotation")
	else:
		fit2D = Minimizer(residual_G2D_norotation,estimates_2D,fcn_args=(x,y),fcn_kws={"data":cropped_data})
		print("Not including rotation")
		
	fit_res2D = fit2D.minimize(minim_method)
	return (x,y,fit_res2D)
	