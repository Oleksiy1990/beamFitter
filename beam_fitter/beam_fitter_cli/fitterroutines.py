import typing
import numpy.typing as npt
import sys
import os.path
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from lmfit import Parameters, minimize, fit_report, Minimizer
import numpy as np

# these are for loading images from files and 
# converting them to numpy arrays
from skimage import io 
from astropy.io import fits

# this is the way to import at the same level inside package
from . import mathmodels

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



import sys, os
import re
import json
import glob

class Image:
    """
    Loads and holds the image from different sources, could be a file, 
    could be a USB camera, for example, or could also be a network connection

    Attributes
    ----------
    data: numpy.ndarray
        Holds the 2D numpy array representing the image
    pixelsize_mm : float or int, default = 1
        Pixel size for the image, expressed in mm
        if the value is 1, this means that the we are working in pixels

    """

    def __init__(self,
                 source: str = "file",
                 pixelsize_mm : typing.Union[float, int] = 1,
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
        pixelsize_mm : float or int, default = 1
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

            # patterns for detecting supported data types
            pat_fits = re.compile(r"\w+\.fits")
            pat_bmp = re.compile(r"\w+\.bmp")
            pat_jpg = re.compile(r"\w+\.jpg")
            pat_png = re.compile(r"\w+\.png")

            # this checks which file extennsion pattern matches 
            is_fits = bool(pat_fits.search(imagepath,re.IGNORECASE))
            is_bmp = bool(pat_bmp.search(imagepath,re.IGNORECASE))
            is_jpg = bool(pat_jpg.search(imagepath,re.IGNORECASE))
            is_png = bool(pat_png.search(imagepath,re.IGNORECASE))
            
            # now we fill the attribute self.data with the appropriate 
            # numpy array
            if is_fits:
                self.data = fits.getdata(imagepath)
            elif is_bmp or is_jpg or is_png:
                self.data = io.imread(imagepath, as_gray=True)
            else:
                print("{:s}.__init__()".
                      format(self.__class__.__name__))
                sys.exit("Unrecognized image format. Exiting")

            if source != "file":
                sys.exit("The image source variable is incorrect. Use file ")
                # this is for possibly making it use pictures directly later

        self.pixelsize_mm = pixelsize_mm

    def crop(self, vertical_low: typing.Union[float,int], 
                    vertical_high: typing.Union[float,int],
                    horizontal_low: typing.Union[float,int],
                    horizontal_high: typing.Union[float,int],
                    measurement_units: str = "pixel") -> None:
        """
        crop the image to a region of interest by giving the edges of the 
        cropped area
        """
        if measurement_units not in ["pixel","mm"]:
            print("{:s}.crop() : Measurement units must be pixel or mm".
                    format(self.__class__.__name__))
            print("you supplied {:s} as measurement_units".format(measurement_units))
            print("This is not allowed, not doing cropping")
            return
        if any(x < 0 for x in [vertical_low,vertical_high,horizontal_low,horizontal_high]):
            print("{:s}.crop() : you supplied the cropping region with at least one negative bound".
                    format(self.__class__.__name__))
            print("This is not allowed, not doing cropping")
            return
        if (vertical_low >= vertical_high) or (horizontal_low >= horizontal_high):
            print("{:s}.crop() : lower bounds of cropping region are equal of above upper bounds".
                    format(self.__class__.__name__))
            print("This is not allowed, not doing cropping")
            
        if measurement_units == "pixel":
            try:
                self.data = self.data[vertical_low:vertical_high,horizontal_low:horizontal_high]
            except IndexError:
                print("{:s}.crop() : cropping region is outside the image indices".
                        format(self.__class__.__name__))
                print("Current image size in pixels: {0}".format(self.data.shape))
                print("Not doing cropping")
        elif measurement_units == "mm":
            vertical_pix_low = int(vertical_low/self.pixelsize_mm)
            vertical_pix_high = int(vertical_high/self.pixelsize_mm)
            horizonal_pix_low = int(horizontal_low/self.pixelsize_mm)
            horizonal_pix_high = int(horizontal_high/self.pixelsize_mm)
            try:
                self.data = self.data[vertical_pix_low:vertical_pix_high,
                                      horizontal_pix_low:horizontal_pix_high]
            except IndexError:
                print("{:s}.crop() : cropping region is outside the image indices".
                        format(self.__class__.__name__))
                print("Current image size in mm: ({:.3f},{:.3f})".
                      format(self.data.shape[0]*self.pixelsize_mm,
                             self.data.shape[1]*self.pixelsize_mm))
                print("Not doing cropping")

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


    def plotimage(self,show=True):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.pcolorfast(self.data)
        if show:
            plt.show()


class Fitter:
    """
    Takes the image and performs a fit
    """
    def __init__(self, image_in : Image) -> None:
        self.image = image_in # this has to hold the image to fit
        # image has to have attributes: 
        #   data -> holding a numpy array, pixelsize_mm -> holding pixel size in mm

        # if it is 1, it means that we are working directly in pixel units 
        self.__pixelsize_to_use = 1

        # vertical_integrated: data summed up along each row
        self.vertical_integrated = self.image.data.sum(axis = 1) 
        self.vertical_xvalues = np.array(range(len(self.vertical_integrated)))
        # horizonal_integrated: data summed up along each column
        self.horizontal_integrated = self.image.data.sum(axis = 0) 
        self.horizontal_xvalues = np.array(range(len(self.horizontal_integrated)))

        # This is for holding arbitrary axes later
        self.arbitrary_axis = None

        # This will hold the image that was passed through the constructor 
        # directly, in order to be able to for example undo cropping later 
        self__original_image = image_in
        
        # the next attributes are private and are used for Gaussian smoothing
        # they should only rarely, if ever, be modified

        self.startparams_vertical = Parameters()
        self.startparams_horizontal = Parameters()
        self.fitresults_dict = {} # will hold a dictionary of what has been 
        # fitted, the values are of type MinimizerResult 


        self.__smoothing_points = 10
        self.__smoothing_sigma_start = 10
        self.__smoothing_sigma_stop = 101
        self.__fraction_for_background = 0.1

    
    def __startparams_estimate(self, arr_in : npt.NDArray) -> typing.Tuple[float, float, float, float]:
        """
        Takes a 1D data array which is supposed to contain a
        Gaussian peak and returns the estimates of peak height, peak position,
        peak width, and background signal. This is used as a starting point for
        1D fitting of Gaussians

        This function is not used on its own, but is only called from fitter functions, which
        feed the data arrays into it

        "smoothened" contains the result of applying a number of Gaussian filter with varying
        widths to the input data array

        NOTE! This function works with the x-coordinate always given only in pixels, there 
        is no conversion to metric units here

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
        # take the max of all maxima from smoothings to estimate peak top
        # NOTE! This is not the same as peak height in a Gaussian formula
        # because the background has to be subtracted for that one
        peak_top_est = np.amax([np.amax(arr) for arr in smoothened])
        # estimate background
        # remember that np.sort() sorts in ascending order
        # we take the number of elements as a fraction of the total length 
        # it computes the mean of the flattened array from all calculated smoothings
        background_est = np.mean([np.sort(arr)[:int(len(arr)*self.__fraction_for_background)] 
                                  for arr in smoothened])
        # estimate peak height
        peak_height_est = peak_top_est - background_est
        
        # estimate peak width
        # note that we are working in this entire function only with pixels, there is no
        # conversion to metric units, and therefore peak width is also given in pixels, 
        # and this is why peak width can be calculated as a fraction of the length of input
        peak_width_est = 0.5*len(arr_in[arr_in > 
                                    peak_height_est/np.power(np.exp(1),2.) + background_est])
        
        # return the estimates; they will be the starting points for fitting    
        # if everything is good, they are nice starting points, leading to fast and 
        # reliable fitting
        return (peak_height_est,peak_pos_est,peak_width_est,background_est)

    def __startparams_estimate_vertical(self) -> None:
        """
        Uses the __startparams_estimate, and applies it directly
        to data summed up along each row, so like integrated 
        image by looking at it from the side

        In this function we will decide whether to work with pixels or mm
        units by using the multiplication conversion factor


        After running this funciton, self.startparams_vertical, which is a Paramters
        object, will contain all the necessary starting parameters to run lmfit
        """
        startparams_vertical_tuple = self.__startparams_estimate(self.vertical_integrated)
        conv = self.__pixelsize_to_use
        # We will now add all the necessary starting parameters to run lmfit
        # Remember that startparams_vertical is a Parameters object
        # NOTE! We do not apply conv (convertsion factor from pixels to mm) 
        # to peak_height, and background
        self.startparams_vertical.add("peak_height",
                                      value = startparams_vertical_tuple[0],
                                      min = 0)
        self.startparams_vertical.add("peak_position",
                                      value = conv*startparams_vertical_tuple[1],
                                      min = conv*0,
                                      max = conv*len(self.vertical_integrated))
        self.startparams_vertical.add("peak_width",
                                      value = conv*startparams_vertical_tuple[2],
                                      min = conv*1,
                                      max = conv*len(self.vertical_integrated))
        self.startparams_vertical.add("background",
                                      value = startparams_vertical_tuple[3],
                                      min = 0,
                                      max = startparams_vertical_tuple[3] + 
                                        startparams_vertical_tuple[0])

    def __startparams_estimate_horizontal(self) -> None:    
        """
        Uses the __startparams_estimate, and applies it directly
        to data summed up along each column, so like integrated 
        image by looking at it from the top
        
        After running this funciton, self.startparams_horizonal, which is a Paramters
        object, will contain all the necessary starting parameters to run lmfit
        """
        startparams_horizontal_tuple = self.__startparams_estimate(self.horizontal_integrated)
        
        conv = self.__pixelsize_to_use
        # We will now add all the necessary starting parameters to run lmfit
        # Remember that startparams_horizonal is a Parameters object
        # NOTE! We do not apply conv (convertsion factor from pixels to mm) 
        # to peak_height, and background
        self.startparams_horizontal.add("peak_height",
                                      value = startparams_horizontal_tuple[0],
                                      min = 0)
        self.startparams_horizontal.add("peak_position",
                                      value=conv*startparams_horizontal_tuple[1],
                                      min = conv*0,
                                      max = conv*len(self.horizontal_integrated))
        self.startparams_horizontal.add("peak_width",
                                      value=conv*startparams_horizontal_tuple[2],
                                      min = conv*1,
                                      max = conv*len(self.horizontal_integrated))
        self.startparams_horizontal.add("background",
                                      value=startparams_horizontal_tuple[3],
                                      min = 0,
                                      max =startparams_horizontal_tuple[3] + 
                                        startparams_horizontal_tuple[0])

    def __fit_integrated_axis(self,
                axis : str,
                minim_method : str = "nelder") -> None:
        """
        This function fits one axis of a 2D array representing an image by doing
        a summation along the other axis

        Parameters
        ----------
        axis : {"vertical", "horizontal"}
        minim_method : str
            This is one of hte minimization methods to be passed 
            to the scipy minimizer working in the background
        """

        if axis not in ["vertical", "horizontal"]:
            sys.exit("""{:s}.__fit_axis(): 
                     axis argument can only have values "vertical" or "horizonal
                     """.format(self.__class__.__name__))
        conv = self.__pixelsize_to_use
        if (axis == "horizontal"):
            axis_data = self.horizontal_integrated
            axis_points = conv*self.horizontal_xvalues
            self.__startparams_estimate_horizontal()
            params_for_fit = self.startparams_horizontal
        elif (axis == "vertical"):
            axis_data = self.vertical_integrated
            axis_points = conv*self.vertical_xvalues
            self.__startparams_estimate_vertical()
            params_for_fit = self.startparams_vertical
        else:
            pass # should not happen because we checked axis argument earlier
        
        # using lmfit package for fitting (based on Scipy)
        # https://lmfit.github.io/lmfit-py/
        fit = Minimizer(mathmodels.residual_G1D,params_for_fit,fcn_args=(axis_points,),\
            fcn_kws={"data":axis_data})
        fit_res = fit.minimize(minim_method)

        if (axis == "horizontal"):
            self.fitresults_dict["horizontal_integrated"] = fit_res 
        elif (axis == "vertical"):
            self.fitresults_dict["vertical_integrated"] = fit_res 
        else:
            pass # this is reserved for other options, like a particular axis
    
    def convert_pixels_to_mm(self, do_conversion : bool = True) -> None:
        """
        Parameters
        ----------
        do_conversion : bool, default = True
            If True, sets the conversion factor to pixel size
            from the image instance, expressed in mm
            if False, sets the conversion factor to 1 (so works
            in pixels themselves)
            Fits have to be rerun if one has changed this option

        Returns
        -------
        None
        """
        if not isinstance(do_conversion, bool):
            print("""Warning: {:s}.convert_pixels_to_mm(): 
                     argument must be a boolean: True or False.
                     Not doing anything.
                     """.format(self.__class__.__name__))
            return
        if do_conversion:     
            self.__pixelsize_to_use = self.image.pixelsize_mm
        else:
            self.__pixelsize_to_use = 1


    def fitandprint_integrated_axis(self,
                                    axis: str,
                                    method : str = "nelder") -> None:

        if axis not in ["vertical", "horizontal"]:
            sys.exit("""{:s}.fitantprint_integrated_axis(): 
                     axis argument can only have values "vertical" or "horizonal"
                     """.format(self.__class__.__name__))
        
        self.__fit_integrated_axis(axis, method)

        if self.__pixelsize_to_use == 1: # this means we are working in pixels
            print("***********************************")
            print("{:s} integrated axis fit results:".format(axis))
            print("positions and lengths in px")
            for (key,val) in self.fitresults_dict["{:s}_integrated".format(axis)].\
                params.valuesdict().items():
                print("{:s} = {:.5f}".format(key,val))

    def fitandplot_integrated_axis(self,
                        axis: str,
                        method : str = "nelder") -> None:
        
        if axis not in ["vertical", "horizontal"]:
            sys.exit("""{:s}.fitantplot_integrated_axis(): 
                     axis argument can only have values "vertical" or "horizonal"
                     """.format(self.__class__.__name__))
        # first we always fit and print the values
        self.fitandprint_integrated_axis(axis, method)
        
        if self.__pixelsize_to_use == 1: 
            position_units = "pixel"
        else:
            position_units = "mm"

        if axis == "vertical":
            xvals = self.vertical_xvalues*self.__pixelsize_to_use
            yvals_data = self.vertical_integrated
        elif axis == "horizontal":
            xvals = self.horizontal_xvalues*self.__pixelsize_to_use
            yvals_data = self.horizontal_integrated
        
        fitparams = self.fitresults_dict["{:s}_integrated".format(axis)].params
        yvals_fit = mathmodels.residual_G1D(fitparams,xvals)
        plt.tight_layout()
        plt.plot(xvals,yvals_data,"ko", label = "Data")
        plt.plot(xvals,yvals_fit,"r--", label = "Fit")
        plt.xlabel("Position [{:s}]".format(position_units))
        plt.ylabel("Intensity [arb. units]")
        plt.title("{:s} axis".format(axis))
        plt.legend()
        plt.show()



# check this, getting packages from the same directory
if __name__ == '__main__':
    # from mathmodels import residual_G1D, residual_G2D, residual_G2D_norotation
    pass
else:
    pass
    # from src.mathmodels import residual_G1D, residual_G2D, residual_G2D_norotation
