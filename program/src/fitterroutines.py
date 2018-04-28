import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from lmfit import Parameters, minimize, fit_report, Minimizer
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from skimage import io 
import sys, os


# check this, getting packages from the same directory
if __name__ == '__main__':
    from mathmodels import residual_G1D, residual_G2D, residual_G2D_norotation
else:
    from src.mathmodels import residual_G1D, residual_G2D, residual_G2D_norotation

class Image:

    def __init__(self,source="file",imagepath=None,pixelsize_mm=5.2e-3,omega_fraction=2):
        """
        get the image from a file or some camera source and convert into into a
        numpy array with skimage.io.imread function

        param: source
        param: imagepath
        param: omega_fraction

        retuns: imagearray, which has the numpy array form of the original image
        """
        if source != "file":
            sys.exit("The image source variable is incorrect. Use file ")
            # this is for possibly making it use pictures directly later

        if source == "file" and isinstance(imagepath,str): 
            try:
                self.image_array = io.imread(imagepath,as_grey=True)
                # image_array is an ndarray as returned by io.imread, in grayscale
            except:
                sys.exit("The image could not be read on path",imagepath)
        else:
            sys.exit("Imagepath parameter must be a string")

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




    def __startparams_estimate(self,data_array):

        """
        Takes a 1D data array which is supposed to contain a
        Gaussian peak and returns the estimates of peak height, peak position,
        peak width, and background signal. This is used as a starting point for
        1D fitting of Gaussians

        This function is not used on its own, but is only called from fitter functions, which
        feed the data arrays into it

        startparams_estimate(data_array)

        "smoothened" applies the Gaussian filter with width from 1/10 to
        1/100 of the data array width, producing 10 arrays. This is done
        in order to get a robust estimate without being thrown off by
        high frequency noise in the data

        """


        # first we smoothen with a Gaussian filter to produce nice estimates of starting
        # fit params
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

        # return the estimates; they will be the starting points for fitting    
        # if everything is good, they are nice starting points, leading to fast and 
        # reliable fitting
        return (peak_height_estim,peak_pos_estim,width_estim,background_estim)


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



    def __fit2D(self,minim_method="nelder",rotation=False):
        
        self.__fit_axis(0,minim_method)
        self.__fit_axis(1,minim_method)

        # we first take all the initial parameters from 1D fits
        bgr2D_est = self.axis0fitparams.valuesdict()["backgr"]/len(self.axis0pts)
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
        estimates_2D.add("theta_rot",value=0*np.pi,min = 0,max = np.pi) #just starting with 0
        estimates_2D.add("backgr",value=bgr2D_est)


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
            
    def fitandprint_2D(self):
        if self.fit2Dparams == None:
            self.__fit2D()
        print("The sizes are in mm")
        for (key,val) in self.fit2Dparams.valuesdict().items():
            print(key,"=",val)

    def fitandplot_2D(self):
        
        if self.fit2Dparams == None:
            self.__fit2D()
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        X,Y = self.x2Dgrid,self.y2Dgrid
        fitted_surface_beam = residual_G2D_norotation(self.fit2Dparams,X,Y)
        original_surface_beam = self.formatted_array
        
        ax.plot_surface(X, Y, fitted_surface_beam, cmap=cm.bwr,\
                       linewidth=0, antialiased=False)
        ax.plot_surface(X,Y,original_surface_beam,cmap=cm.bwr,linewidth=0,antialiased=False)

        #cset = ax.contour(X, Y, fitted_surface_beam, zdir='z',offset=0, cmap=cm.bwr)
        cset = ax.contour(X, Y, fitted_surface_beam, zdir='x',\
            offset=X[0], cmap=cm.bwr)
        cset = ax.contour(X, Y, fitted_surface_beam, zdir='y',\
            offset=Y[-1], cmap=cm.bwr)

        #cset1 = ax.contour(X, Y, original_surface_beam, zdir='z',offset=0, cmap=cm.bwr)
        cset1 = ax.contour(X, Y, original_surface_beam, zdir='x',\
            offset=X[0], cmap=cm.bwr)
        cset1 = ax.contour(X, Y, original_surface_beam, zdir='y',\
            offset=Y[-1], cmap=cm.bwr)
        #fig.colorbar(surf)

        plt.show()
        

    def plotimage(self,show=True):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.pcolorfast(self.image_array)
        if show:
            plt.show()

if __name__ == '__main__':
    q = Image(source="file",imagepath="Q:\\qgases\\groups\\strontium\\Oleksiy\\transm1602\\7.bmp")
    #q = Image(source="file",imagepath="/Users/oleksiy/Desktop/PythonCode/beamFitter/ExampleImages/framebefore1.bmp")
    #q.plotimage()
    #print(q.fit_axis(0))
    #q.fitandprint_2D()
    #q.fitandplot_2D()
    q.fitandprint_axis(0)
    q.fitandplot_axis(0)
    q.fitandprint_axis(1)
    q.fitandplot_axis(1)

    #Axis 0 is vertical
    #Axis 1 is horizontal

