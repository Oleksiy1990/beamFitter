import typing
import numpy.typing as npt
import sys
import os.path
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from lmfit import Parameters, minimize, fit_report, Minimizer
from skimage import io 
import sys, os
from astropy.io import fits
import re
import json
import glob


class Fitter:
    """
    Takes the image and performs a fit
    """
    def __init__(self) -> None:
        self.image = None # this has to hold the image to fit
        # image has to have attributes: 
        #   data -> holding a numpy array, pixelsize_mm -> holding pixel size in mm
        
        # the next attributes are private and are used for Gaussian smoothing
        # they should only rarely, if ever, be modified

        self.peak_pos_estimate = np.array([-1.,-1.]) # -1 means that it is not calculated yet

        self.__smoothing_points = 10
        self.__smoothing_sigma_start = 10
        self.__smoothing_sigma_stop = 101
        self.__fraction_for_background = 0.1

    def __startparams_estimate(self, arr_in : npt.NDArray):

        """
        Takes a 1D data array which is supposed to contain a
        Gaussian peak and returns the estimates of peak height, peak position,
        peak width, and background signal. This is used as a starting point for
        1D fitting of Gaussians

        This function is not used on its own, but is only called from fitter functions, which
        feed the data arrays into it

        "smoothened" contains the result of applying a number of Gaussian filter with varying
        widths to the input data array

        Parameters
        ----------
        arr_in : ndarray
            1D array of data to have Gaussian fit parameters estimated

        Returns
        -------

        """

        gaussian_sigma_list = list(range(self.__smoothing_sigma_start,
                                         self.__smoothing_sigma_stop,
                                         self.__smoothing_points))
        smoothened = [gaussian_filter1d(arr_in,
                                    len(arr_in)/x) 
                                    for x in gaussian_sigma_list]
        # take the mean of max locations from smoothings to estimate peak position
        peak_pos_est = np.mean([np.argmax(arr) for arr in smoothened]) 
        # take the max of all maxima from smoothings to estimate peak height
        peak_height_est = np.amax([np.amax(arr) for arr in smoothened])
        # estimate background
        # remember that np.sort() sorts in ascending order
        # we take the number of elements as a fraction of the total length 
        # it computes the mean of the flattened array from all calculated smoothings
        background_est = np.mean([np.sort(arr)[:int(len(arr)*self.__fraction_for_background)] 
                                  for arr in smoothened])
        
        # the estimation of the peak width only takes into account the most
        # highly smoothened data in this case
        
        
        # return the estimates; they will be the starting points for fitting    
        # if everything is good, they are nice starting points, leading to fast and 
        # reliable fitting
        return (peak_height_estim,peak_pos_estim,width_estim,background_estim)
    def __startparams_estimate(self):



        data_array_vertical = self.image.data.sum(axis = 1) # we sum up along each row,
        # so the values of different columns for each row are summed up
        # this is like looking at the image from the side

        data_array_horizontal = self.image.data.sum(axis = 0) # we sum up along each column,
        # so the values of different rows for each column are summed up
        # this is like looking at the image from the top

        # first we smoothen with a Gaussian filter to produce nice estimates of starting
        # fit params
        smoothened_vertical = [gaussian_filter1d(data_array_vertical,
                                    len(data_array_vertical)/x) 
                                    for x in range(self.__smoothing_sigma_start,
                                                    self.__smoothing_sigma_stop,
                                                    self.__smoothing_points)]
        smoothened_horizontal = [gaussian_filter1d(data_array_horizontal,
                                    len(data_array_horizontal)/x) 
                                    for x in range(self.__smoothing_sigma_start,
                                                    self.__smoothing_sigma_stop,
                                                    self.__smoothing_points)]
        
        # we take averages of the estimates based on data with different
        # amounts of smoothing
        peak_pos_horizonal = np.argmax(smoothened,axis=1)
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

        # return the estimates; they will be the starting points for fitting    
        # if everything is good, they are nice starting points, leading to fast and 
        # reliable fitting
        return (peak_height_estim,peak_pos_estim,width_estim,background_estim)


class Image:
    """
    Loads and holds the image from different sources, could be a file, 
    could be a USB camera, for example, or could also be a network connection

    Attributes
    ----------


    """

    def __init__(self,
                 source: str = "file",
                 imagepath : str = "",
                 comport : str = "",
                 debug : bool = False):
                 #,pixelsize_mm=5.2e-3,omega_fraction=2):
        """
        get the image from a file or some camera source and convert into into a
        numpy array with skimage.io.imread function

        Parameters
        ----------
        source : {"file","COMPORT","LAN"}
            When getting the images from file, one has to write imagepath,
            when getting the images from the comport, one has to use the appropriate
            serial communication approach, and specify COM-port and baud rate,
            when getting the images from LAN, one also has to appropriately 
            specify IP address, etc.
            
            LAN not implemented yet
        imagepath : str, default = ""
            either the full path to the image, or the relative path with 
            respect to the current working directory
        comport: str, default = ""
        debug : bool, default = False


        param: source
        param: imagepath
        param: omega_fraction

        retuns: imagearray, which has the numpy array form of the original image
        """


        if source == "file":
            # write the methods here to generate a numpy array when the 
            # image is to be taken from a file
            if debug:
                print("{:s}.__init__(): imagepath provided: {:s}".
                      format(self.__class__.__name__,imagepath))
            if not os.path.isfile(imagepath):
                print("{:s}.__init__(): imagepath provided: {:s}".
                      format(self.__class__.__name__,imagepath))
                sys.exit("Wrong path to file, exiting")
                # if the image path is wrong, we exit the application (for now)
            self.pathbase = os.path.splitext(imagepath)[0]
            print("Base path: ",self.pathbase)

            pat_fits = re.compile(r"\w+\.fits")
            pat_bmp = re.compile(r"\w+\.bmp")
            pat_jpg = re.compile(r"\w+\.jpg")
            pat_png = re.compile(r"\w+\.png")
            is_fits = bool(pat_fits.search(imagefile,re.IGNORECASE))
            is_bmp = bool(pat_bmp.search(imagefile,re.IGNORECASE))
            is_jpg = bool(pat_jpg.search(imagefile,re.IGNORECASE))
            is_png = bool(pat_png.search(imagefile,re.IGNORECASE))
            if is_fits:
                self.image_array = fits.getdata(imagepath)
            elif is_bmp or is_jpg or is_png:
                self.image_array = io.imread(imagepath, as_grey=True)
            else:
                print("{:s}.__init__()".
                      format(self.__class__.__name__))
                sys.exit("Unrecognized image format. Exiting")

            if source != "file":
                sys.exit("The image source variable is incorrect. Use file ")
                # this is for possibly making it use pictures directly later

        self.__pixelsize = pixelsize_mm
        self.__omega_fraction = omega_fraction

        self.axis0pts = None
        self.axis0data = None
        self.axis0fitresult = None
        self.axis1pts = None
        self.axis1data = None
        self.axis1fitresult = None

        self.x2Dgrid = None
        self.y2Dgrid = None
        self.fit2Dparams = None

        self.formatted_array = None #This will hold the array that's cut out of the picture


    def __format_picture(self,x_cent,omega_x,y_cent,omega_y):

        """
        This function takes a 2D array and the values for center position of a Gaussian
        peak and its width to crop out a region within a given number of omega from the
        center (the default is 2*omega). This is useful for 2D fitting and plotting, when
        the beam is much smaller than the entire picture size, it's much faster to
        only fit where the peak as, without evaluating too many background points

        """

        # one has to be careful here because index 0 is not necessarily "x" in plots
        # index 0 is always rows, and that's what usually plotted vertically

        length_x = self.image_array.shape[0]
        length_y = self.image_array.shape[1]

        # Define the positions of the borders on left, right, top, bottom 
        b_left = int(x_cent - self.__omega_fraction*omega_x)
        b_right = int(x_cent + self.__omega_fraction*omega_x)
        b_bottom = int(y_cent - self.__omega_fraction*omega_y)
        b_top = int(y_cent + self.__omega_fraction*omega_y)
        b_left = b_left if (b_left >= 0) else 0
        b_right = b_right if (b_right <= length_x) else length_x
        b_bottom = b_bottom if (b_bottom >= 0) else 0
        b_top = b_top if (b_top <= length_y) else length_y


        self.formatted_array = self.image_array[b_left:b_right,b_bottom:b_top]


    def __fit_axis(self,axis,minim_method="nelder"):
        """
        This function fits one axis of a 2D array representing an image by doing
        a summation along the other axis

        fit_axis(axis,minim_method="nelder")
        """

        if axis not in [0,1]:
            sys.exit("The axis can only be 0 or 1 in fit_axis function")


        axis_data = np.sum(self.image_array,axis = 1) if (axis == 0) else np.sum(self.image_array,axis = 0)
        axis_points = np.linspace(1,len(axis_data),len(axis_data))
        param_estimates = self.__startparams_estimate(axis_data)

        # using lmfit package for fitting (based on Scipy)
        # https://lmfit.github.io/lmfit-py/
        params_for_fit = Parameters()
        params_for_fit.add('I_zero',value=param_estimates[0],min=0,max=np.amax(axis_data))
        params_for_fit.add('r_zero',value=param_estimates[1],min=1,max=len(axis_data))
        params_for_fit.add('omega_zero',value=param_estimates[2],min=1,max=len(axis_data))
        params_for_fit.add('backgr',value=param_estimates[3])
        fit = Minimizer(residual_G1D,params_for_fit,fcn_args=(axis_points,),\
            fcn_kws={"data":axis_data})
        fit_res = fit.minimize(minim_method)

        if axis == 0:
            self.axis0pts = axis_points
            self.axis0data = axis_data
            self.axis0fitparams = fit_res.params

        elif axis == 1:
            self.axis1pts = axis_points
            self.axis1data = axis_data
            self.axis1fitparams = fit_res.params

        # return (axis_points,axis_data,fit_res)



    def __fit2D(self,minim_method="leastsq",rotation=False,initial_params_2D = None):
        if initial_params_2D is None:
            self.__fit_axis(0,minim_method)
            self.__fit_axis(1,minim_method)

            # we first take all the initial parameters from 1D fits
            bgr2D_est = np.mean([self.axis0fitparams.valuesdict()["backgr"]/len(self.axis0pts),self.axis1fitparams.valuesdict()["backgr"]/len(self.axis1pts)])
            x2D_est = self.axis0fitparams.valuesdict()["r_zero"]
            omegaX2D_est = self.axis0fitparams.valuesdict()["omega_zero"]
            y2D_est = self.axis1fitparams.valuesdict()["r_zero"]
            omegaY2D_est = self.axis1fitparams.valuesdict()["omega_zero"]

            smoothened_image = gaussian_filter(self.image_array,50)
            peakheight2D_est = np.amax(smoothened_image)
            #now we need to programatically cut out the region of interest out of the
            #whole picture so that fitting takes way less time

            # NOTE! In this implementation, if the beam is small compared to picture size
            # and is very close to the edge, the fitting will fail, because the x and y
            # center position estimates will be off

            self.__format_picture(x2D_est,omegaX2D_est,y2D_est,omegaY2D_est)
            cropped_data = self.formatted_array
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
            estimates_2D.add("theta_rot",value=0.578,min = 0,max = np.pi/2) #just starting with 0
            estimates_2D.add("backgr",value=bgr2D_est)
            print("Here are the parameters before the start of the fit: ")
            print(estimates_2D.valuesdict())
        else:
            estimates_2D = initial_params_2D
            print("Here are the parameters before the start of the fit: ")
            print(estimates_2D.valuesdict())

        if rotation:
                fit2D = Minimizer(residual_G2D,estimates_2D,fcn_args=(x,y),fcn_kws={"data":cropped_data})
                print("Including rotation")
        else:
            fit2D = Minimizer(residual_G2D_norotation,estimates_2D,fcn_args=(x,y),fcn_kws={"data":cropped_data})
            print("Not including rotation")

        fit_res2D = fit2D.minimize(minim_method)

        self.x2Dgrid = x
        self.y2Dgrid = y
        self.fit2Dparams = fit_res2D.params
        # return (x,y,fit_res2D)

    def fitandprint_axis(self,axis):

        if axis == 0:
            if self.axis0fitresult == None:
                self.__fit_axis(axis)
            #print("The sizes are in mm")
            for (key,val) in self.axis0fitparams.valuesdict().items():
                if key in ["I_zero","backgr"]:
                    print(key,"=",val)
                elif key in ["r_zero","omega_zero"]:
                    print(key,"=",val*self.__pixelsize,"mm")
                else:
                    print("something went wrong in fitandprint_axis")
        if axis == 1:
            if self.axis1fitresult == None:
                self.__fit_axis(axis)
            print("The sizes are in mm")
            for (key,val) in self.axis1fitparams.valuesdict().items():
                if key in ["I_zero","backgr"]:
                    print(key,"=",val)
                elif key in ["r_zero","omega_zero"]:
                    print(key,"=",val*self.__pixelsize,"mm")
                else:
                    print("something went wrong in fitandprint_axis")

    def fitandplot_axis(self,axis):
        
        if axis == 0:
            if self.axis0fitresult == None:
                self.__fit_axis(axis)
            plt.plot(self.axis0pts*self.__pixelsize,self.axis0data,"r-",\
                self.axis0pts*self.__pixelsize,residual_G1D(self.axis0fitparams,self.axis0pts),"k-")
            plt.xlabel("Position (mm)")
            plt.ylabel("Intensity (arb. units)")
            plt.title("Axis %.i"%axis)
            plt.legend(("Data","Fit"))
            plt.show()

        if axis == 1:
            if self.axis1fitresult == None:
                self.__fit_axis(axis)
            plt.plot(self.axis1pts*self.__pixelsize,self.axis1data,"r-",\
                self.axis1pts*self.__pixelsize,residual_G1D(self.axis1fitparams,self.axis1pts),"k-")
            plt.xlabel("Position (mm)")
            plt.ylabel("Intensity (arb. units)")
            plt.title("Axis %.i"%axis)
            plt.legend(("Data","Fit"))
            plt.show()
            
    def fitandprint_2D(self,rotation=False,save_json = False):
        if self.fit2Dparams == None:
            self.__fit2D(rotation=rotation)
        print("The sizes are in px")
        if save_json:
            json_file = self.pathbase+".json"
            with open(json_file,"w") as outfile:
                json.dump(self.fit2Dparams.valuesdict(),outfile,indent=4)
        for (key,val) in self.fit2Dparams.valuesdict().items():
            print(key,"=",val)

    def fitandplot_2D(self,rotation=False):
        
        if self.fit2Dparams == None:
            self.__fit2D(rotation=rotation)
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        X,Y = self.x2Dgrid,self.y2Dgrid
        if rotation:
            fitted_surface_beam = residual_G2D(self.fit2Dparams,X,Y)
        else:
            fitted_surface_beam = residual_G2D_norotation(self.fit2Dparams,X,Y)
        original_surface_beam = self.formatted_array
        
        ax.plot_surface(X, Y, fitted_surface_beam, cmap=cm.bwr,\
                       linewidth=0, antialiased=False)
        ax.plot_surface(X,Y,original_surface_beam,cmap=cm.bwr,linewidth=0,antialiased=False)

        #cset = ax.contour(X, Y, fitted_surface_beam, zdir='z',offset=0, cmap=cm.bwr)
        cset = ax.contour(X, Y, fitted_surface_beam, zdir='x',\
            offset=0, cmap=cm.bwr)
        cset = ax.contour(X, Y, fitted_surface_beam, zdir='y',\
            offset=0, cmap=cm.bwr)

        #cset1 = ax.contour(X, Y, original_surface_beam, zdir='z',offset=0, cmap=cm.bwr)
        cset1 = ax.contour(X, Y, original_surface_beam, zdir='x',\
            offset=0, cmap=cm.bwr)
        cset1 = ax.contour(X, Y, original_surface_beam, zdir='y',\
            offset=0, cmap=cm.bwr)
        #fig.colorbar(surf)

        plt.show()
        

    def plotimage(self,show=True):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.pcolorfast(self.image_array)
        if show:
            plt.show()

if __name__ == '__main__':
    omega_frac = 3
    imgpath = "D:\\VU\\HeNeImg\\"
    filelist = glob.glob(imgpath+"*.fits")
    fnames = [os.path.split(file)[1] for file in filelist]
    print(fnames)
    #fname = "10holes.fits"
    for fname in fnames:
        print(fname)
        q = Image(source="file",imagepath=imgpath,imagefile=fname,pixelsize_mm=5.3e-3,omega_fraction=omega_frac)
        #q.plotimage()
        #q.fitandprint_axis(0)
        #q.fitandplot_axis(0)
        #q.fitandprint_axis(1)
        #q.fitandplot_axis(1)
        q.fitandprint_2D(rotation=True,save_json=True)
        q.fitandplot_2D(rotation=True)

    #Axis 0 is vertical
    #Axis 1 is horizontal


# check this, getting packages from the same directory
if __name__ == '__main__':
    from mathmodels import residual_G1D, residual_G2D, residual_G2D_norotation
else:
    from src.mathmodels import residual_G1D, residual_G2D, residual_G2D_norotation
