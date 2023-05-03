"""
This file defined the math models used in fitting beams. These are the models
that go into the Minimizer function of the lmfit package to carry out minimization

As of now, there are only two models, 1D and 2D Gaussians, corresponding to 
Gaussian 00 mode in 2D, but this can possibly be extended later for more 
complicated beam patterns, like the non-00 mode
"""


import numpy as np 
import lmfit 
import typing
import numpy.typing as npt

def residual_G1D(
		pars : lmfit.Parameters, 
		x : npt.NDArray, 
		data : typing.Union[npt.NDArray,None] = None, 
		eps : typing.Union[npt.NDArray,None] = None):
	""" 
	G1D stands for "Gaussian 1D" 

	this function defines the Gaussian model for beam fitting and
	returns the residuals to be minimized if data are supplied, or 
	the Gaussian model itself if no data are supplied

	residual_G1D(pars, x, data=None, eps=None)

	Parameters
	----
    pars : lmfit.Parameters object
        This object must contain the initial guesses for the parameters 
    x : array_like
        the x-values of the data
    data : array_like or None, optional, default = None
        the data to be fit
    eps : float or array_like or None, optional, default = None
        the error bars in the data

    Returns
    -------
    array_like
        the residuals to be minimized
	"""
	parvals = pars.valuesdict() # a Parameters() object is passed as "pars"
	intensity_max = parvals["peak_height"]
	centerposition = parvals["peak_position"]
	beamwidth = parvals["peak_width"]
	bgr = parvals["background"]
	model = intensity_max*np.exp(-2.0*np.power(x-centerposition,2)/beamwidth**2) + bgr
	if data is None:
		return np.array(model)
	if eps is None:
		return np.array(model - data)
	return np.array((model - data)/eps)

def residual_G2D(pars, x, y, data = None, eps = None):
	"""
	G2D stands for "Gaussian 2D"
	NOT IMPLEMENTED YET

	this function defines the Gaussian model for beam fitting in 2D and
	returns the residuals to be minimized if data are supplied, or 
	the Gaussian model itself if no data are supplied. The residuals are 
	returned in as a flattened list, because that's how LMFIT wants them. 
	If the model itself is returned, the list is not flattened

	residual_G2D(pars,x,y,data=None, eps=None)
	"""
	parvals = pars.valuesdict() # a Parameters() object is passed as "pars"
	intensity_max = parvals["I_zero"]
	centerposition_x = parvals["x_zero"]
	centerposition_y = parvals["y_zero"]
	beamwidth_x = parvals["omegaX_zero"]
	beamwidth_y = parvals["omegaY_zero"]
	theta = parvals["theta_rot"]
	bgr = parvals["backgr"]
	
	# the model function is based on this http://www.cs.brandeis.edu/~cs155/Lecture_06.pdf
	# slide 23; it should describe rotation by angle theta around an arbitrary point 
	# if I understood the notes correctly, then this transformation should be correct 
	# but I didn't check the math myself

	# the rotation is clockwise

	model = intensity_max*np.exp(-2*np.power(x*np.cos(theta)-y*np.sin(theta)+centerposition_x*(1-np.cos(theta))+centerposition_y*np.sin(theta)-centerposition_x,2)/beamwidth_x**2 - \
		2*np.power(x*np.sin(theta)+y*np.cos(theta)+centerposition_y*(1-np.cos(theta))-centerposition_x*np.sin(theta)-centerposition_y,2)/beamwidth_y**2) + bgr
	if data is None:
		return np.array(model) # we don't flatten here because this is for plotting
	if eps is None:
		resid = np.array(model - data)
		return resid.flatten() # minimization array must be flattened (LMFIT FAQ)
	else:
		resid = np.array((model - data)/eps)
		return resid.flatten()

def residual_G2D_norotation(pars,x,y,data=None, eps=None):
	"""
	NOT IMPLEMENTED YET
	G2D stands for "Gaussian 2D"
	This model assumes that the elliptical Gaussian beam is aligned parallel
	to the axes, so there's no rotation angle theta

	this function defines the Gaussian model for beam fitting in 2D and
	returns the residuals to be minimized if data are supplied, or 
	the Gaussian model itself if no data are supplied. The residuals are 
	returned in as a flattened list, because that's how LMFIT wants them. 
	If the model itself is returned, the list is not flattened

	residual_2D(pars,x,y,data=None, eps=None)
	"""
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