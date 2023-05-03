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

class UnknownFileExtensionError(Exception):
    """
    This exception is raised when an unknown file extension is used
    """
    def __init__(self, image_path : str) -> None:
        self.message = """
            Provided image path: {:s} . Such a file extension is not supported.
            """.format(image_path)
        super().__init__(self.message)


class UnknownImageSourceError(Exception):
    """
    This exception is raised when an unknown file extension is used
    """
    def __init__(self, source : str) -> None:
        self.message = """
            Provided image source: {:s} . Such a source is not supported.
            """.format(source)
        super().__init__(self.message)


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
                 imagepath : str = "",
                 pixelsize_mm : typing.Union[float, int] = 1,
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

        Attributes
        ----------
        data: numpy.ndarray
        pixelsize_mm : float or int, default = 1

        """

        # Accessible from the outside with @property
        self.__data = None
        self.__pixelsize_mm = pixelsize_mm
        self.__crop_vertical_offset_pixels = 0
        self.__crop_horizontal_offset_pixels = 0

        # Private, should not be accessed from outside
        self.__imagepath = imagepath
        self.__image_file_extension = ""

        if source == "file":
            if debug:
                print("{:s}.__init__(): imagepath provided: {:s}".
                      format(self.__class__.__name__,self.__imagepath))
            self.__check_if_image_exists()
            self.__parse_image_extension()
            self.__load_image_as_numpy_array()
        else:
            raise UnknownImageSourceError(source)
        
    @property
    def data(self) -> np.ndarray:
        return self.__data

    @property
    def pixelsize_mm(self) -> typing.Union[int, float]:
        return self.__pixelsize_mm
    @pixelsize_mm.setter
    def pixelsize_mm(self, pixelsize_mm : typing.Union[int, float]): 
        if not isinstance(pixelsize_mm, (float, int)):
            print("{:s}.__set_pixelsize_mm(): pixelsize_mm provided: {:s}".
                    format(self.__class__.__name__,pixelsize_mm))
            raise TypeError("pixelsize_mm must be float or int")
        if pixelsize_mm < 0:
            print("{:s}.__set_pixelsize_mm(): pixelsize_mm provided: {:s}".
                    format(self.__class__.__name__,pixelsize_mm))
            raise ValueError("pixelsize_mm must be positive")
        if pixelsize_mm > 1.0:
            print("{:s}.__set_pixelsize_mm(): pixelsize_mm provided: {:s}".
                    format(self.__class__.__name__,pixelsize_mm))
            print("Warning: pixelsize_mm is greater than 1.0")
            print("Are you sure that you are expressing is in mm?")
        self.__pixelsize_mm = pixelsize_mm

    @property
    def crop_vertical_offset(self) -> typing.Union[int, np.integer]:
        return self.__crop_vertical_offset_pixels

    @property
    def crop_horizonal_offset(self) -> typing.Union[int, np.integer]:
        return self.__crop_horizontal_offset_pixels
    
    def __check_if_image_exists(self) -> None:
        if not os.path.isfile(self.__imagepath):
            print("Class {:s}: imagepath provided: {:s}".
                    format(self.__class__.__name__,self.__imagepath))
            raise FileNotFoundError("This image file cannot be found")

    def __parse_image_extension(self):
        # patterns for detecting supported data types
        pat_fits = re.compile(r"\w+\.fits",re.IGNORECASE)
        pat_bmp = re.compile(r"\w+\.bmp",re.IGNORECASE)
        pat_jpg = re.compile(r"\w+\.jpg",re.IGNORECASE)
        pat_png = re.compile(r"\w+\.png",re.IGNORECASE)

        if bool(pat_fits.search(self.__imagepath)):
            self.__image_file_extension = "fits"
        elif bool(pat_bmp.search(self.__imagepath)):
            self.__image_file_extension = "bmp"
        elif bool(pat_jpg.search(self.__imagepath)):
            self.__image_file_extension = "jpg"
        elif bool(pat_png.search(self.__imagepath)):
            self.__image_file_extension = "png"
        else:
            print("{:s}.__parse_image_extensions(): imagepath provided: {:s}".
                    format(self.__class__.__name__,self.__imagepath))
            print("Provided extension is not supported")

    def __load_image_as_numpy_array(self) -> None:
        if self.__image_file_extension == "fits":
            self.__data = fits.getdata(self.__imagepath)
        elif self.__image_file_extension in ["bmp","jpg","png"]:
            self.__data = io.imread(self.__imagepath, as_gray=True)
        else:
            print("{:s}.__init__()".
                    format(self.__class__.__name__))
            raise UnknownFileExtensionError(self.__imagepath)

    def crop_in_pixels(self, 
                    vertical_low: typing.Union[float,int], 
                    vertical_high: typing.Union[float,int],
                    horizontal_low: typing.Union[float,int],
                    horizontal_high: typing.Union[float,int]
                    ):
        input_params = [vertical_low,vertical_high,horizontal_low,horizontal_high]
        """

        Crop the image to a region of interest by giving the edges of the 
        cropped area
        
        This is the main function for cropping, that also checks if the 
        bounds are integers, all non-negative, and arranged well (this means
        that high bound is greater than low bound for both horizonal and 
        vertical directions)
        """
        if self.__are_bounds_integers(*input_params) and \
                self.__are_all_bounds_nonnegative(*input_params) and \
                self.__are_bounds_arranged_well(*input_params):
            self.__data = self.__data[vertical_low:vertical_high,horizontal_low:horizontal_high]
            self.__crop_vertical_offset_pixels = vertical_low
            self.__crop_horizontal_offset_pixels = horizontal_low
        else:
            print("{:s}.crop_in_pixels(): Cropping failed, see previous error messages".
                    format(self.__class__.__name__))

    def crop_in_mm(self, 
                    vertical_low: typing.Union[float,int], 
                    vertical_high: typing.Union[float,int],
                    horizontal_low: typing.Union[float,int],
                    horizontal_high: typing.Union[float,int]):
        input_params_in_mm = [vertical_low,vertical_high,horizontal_low,horizontal_high]
        if not self.__are_all_bounds_nonnegative(*input_params_in_mm):
            return False
        input_params_in_pixels = self.__convert_mm_to_pixels(input_params_in_mm)
        self.crop_in_pixels(*input_params_in_pixels)
    
    def crop_in_pixels_centered(self, 
                    vertical_center: typing.Union[float,int], 
                    vertical_halfwidth: typing.Union[float,int],
                    horizontal_center: typing.Union[float,int],
                    horizontal_halfwidth: typing.Union[float,int]
                    ) -> None:
        """
        crop the image to a region of interest by giving the center and halfwidth 
        of the cropped area
        """
        input_params_centered = [vertical_center,vertical_halfwidth,
                                 horizontal_center,horizontal_halfwidth]
        if not self.__are_all_bounds_nonnegative(*input_params_centered):
            return False
        crop_bounds = self.__generate_crop_bounds_for_centered(*input_params_centered)
        self.crop_in_pixels(*crop_bounds)
    
    def crop_in_mm_centered(self, 
                    vertical_center: typing.Union[float,int], 
                    vertical_halfwidth: typing.Union[float,int],
                    horizontal_center: typing.Union[float,int],
                    horizontal_halfwidth: typing.Union[float,int]
                    ) -> None:
        """
        crop the image to a region of interest by giving the center and halfwidth 
        of the cropped area
        """
        crop_region_in_mm_centered = [vertical_center,vertical_halfwidth,
                                 horizontal_center,horizontal_halfwidth]
        if not self.__are_all_bounds_nonnegative(*crop_region_in_mm_centered):
            return False
        crop_region_in_pixels_centered = self.__convert_mm_to_pixels(crop_region_in_mm_centered) 
        crop_bounds = self.__generate_crop_bounds_for_centered(*crop_region_in_pixels_centered)
        self.crop_in_pixels(*crop_bounds)

    def __are_bounds_integers(self,
                    vertical_low: typing.Union[float,int], 
                    vertical_high: typing.Union[float,int],
                    horizontal_low: typing.Union[float,int],
                    horizontal_high: typing.Union[float,int]
                    ) -> bool:
        input_params = [vertical_low,vertical_high,horizontal_low,horizontal_high]
        if all(isinstance(x,(int,np.integer)) for x in input_params):
            return True
        else:
            print("""{:s}.__are_all_bounds_integers() : supplied bounds are {} 
                Types of supplied bounds are: {}
                Some supplied bounds are not integers""".
                    format(self.__class__.__name__, input_params,
                           list(map(lambda x: type(x), input_params))))
            print("This is not allowed, not doing cropping")
            return False

    def __are_all_bounds_nonnegative(self,
                    vertical_low: typing.Union[float,int], 
                    vertical_high: typing.Union[float,int],
                    horizontal_low: typing.Union[float,int],
                    horizontal_high: typing.Union[float,int]
                    ) -> bool:
        if any(x < 0 for x in [vertical_low,vertical_high,horizontal_low,horizontal_high]):
            print("""{:s}.__are_all_bounds_nonnegative() 
                you supplied the cropping region with at least one negative bound""".
                    format(self.__class__.__name__))
            print("This is not allowed, not doing cropping")
            return False
        else:
            return True
        
    def __are_bounds_arranged_well(self,
                    vertical_low: typing.Union[float,int], 
                    vertical_high: typing.Union[float,int],
                    horizontal_low: typing.Union[float,int],
                    horizontal_high: typing.Union[float,int]
                    ) -> bool:
        if (vertical_low >= vertical_high) or (horizontal_low >= horizontal_high):
            print("""{:s}.__are_all_bounds_arranged_well() : 
                lower bounds of cropping region are equal of above upper bounds""".
                    format(self.__class__.__name__))
            print("This is not allowed, not doing cropping")
            return False
        else: 
            return True
    
    def __convert_mm_to_pixels(self, values_in : typing.List) -> typing.List[int]:
        return list(map(lambda x: int(x/self.pixelsize_mm), values_in))

    def __generate_crop_bounds_for_centered(self,
                    vertical_center: int, 
                    vertical_halfwidth: int,
                    horizontal_center: int,
                    horizontal_halfwidth: int) -> typing.List[int]:
        generated_crop_region = [
                    np.clip(vertical_center - vertical_halfwidth,0,None),
                    vertical_center + vertical_halfwidth + 1,
                    np.clip(horizontal_center - horizontal_halfwidth,0,None),
                    horizontal_center + horizontal_halfwidth + 1
                    ]
        return generated_crop_region    

    def plotimage(self,show=True):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.pcolorfast(self.__data)
        if show:
            plt.show()
