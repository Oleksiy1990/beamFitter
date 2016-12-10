import sys
from PyQt4 import QtGui, QtCore
from mainwindow import Ui_MainWindow
import time

#sys.path.append('C:\\Users\\Oleksiy\\Desktop\\Code\\beamFitter')

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

from lmfit import Parameters, minimize, fit_report
import numpy as np



import src.imageUSB as imggrab
import src.fitterroutines as frt
import src.mathmodels as mm 


class MyWindowClass(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.timerfunc)

        QtCore.QObject.connect(self.ui.StartButton, QtCore.SIGNAL('clicked()'), self.starter)
        #self.timer.connect(self.ui.StartButton, QtCore.SIGNAL('clicked()'), self.doFitting_stream)
        QtCore.QObject.connect(self.ui.fitFromFile, QtCore.SIGNAL('clicked()'), self.doFitting_file)
        QtCore.QObject.connect(self.ui.fileDialog, QtCore.SIGNAL('clicked()'), self.showFileInput)
        QtCore.QObject.connect(self.ui.PauseButton, QtCore.SIGNAL('clicked()'), self.close)


    def starter(self):
        self.timer.start(1000)

    def timerfunc(self):
        self.ui.ErrorMessages.append(time.asctime( time.localtime(time.time())))
        


    
    # def fitting(self,data):
    #     """
    #     This function does the fitting along the 2 axes independently (using integrated  
    #     images along axes), and it can be extended to do the full 2D fitting.

    #     The "data" input is the 2D numpy array representing the image. The idea is that 
    #     this can be anything, most importantly the data coming from an Ethernet cam

    #     """
    #     print("Doing fitting")
    #     #print(data.shape)
    #     scale_pixels_mm = self.ui.pixelSizeInput.value()*10e-3
    
    #     # Do the fits along each axis. Try if it works and show an error message if it fails
    #     try:
    #         fit0 = frt.fit_axis(data,1) # NOTE! Axes and array indexing are messed up
    #         fit1 = frt.fit_axis(data,0)
    #     except:
    #         self.ui.FitResultsHorizontal.setPlainText("")
    #         self.ui.FitResultsVertical.setPlainText("")
    #         self.ui.ErrorMessages.setPlainText("Fitting failed!")

    #         #canvas_forbeam = self.ui.BeamDisplay.canvas
    #         #canvas_forbeam.ax.pcolorfast(data)
    #         #canvas_forbeam.draw()

    #         return None #if there's an error, we stop the procedure
           
        
    #     # Read out and display fit parameters along horiz. axis
    #     fitted_params_horiz = fit0[2].params.valuesdict()
    #     axis0_pos = fitted_params_horiz["r_zero"]*scale_pixels_mm
    #     axis0_omega = fitted_params_horiz["omega_zero"]*scale_pixels_mm
    #     self.ui.FitResultsHorizontal.setPlainText("")
    #     self.ui.FitResultsHorizontal.append("Units: mm")
    #     self.ui.FitResultsHorizontal.append("Peak position %.3f" % axis0_pos)
    #     self.ui.FitResultsHorizontal.append("Beam width %.3f" % axis0_omega)
    #     #self.ui.FitResultsHorizontal.append("Height above bgr %.3f" % fitted_params_horiz["I_zero"])

    #     # Read out and display fit parameters along horiz. axis
    #     fitted_params_vert = fit1[2].params.valuesdict()
    #     axis1_pos = fitted_params_vert["r_zero"]*scale_pixels_mm
    #     axis1_omega = fitted_params_vert["omega_zero"]*scale_pixels_mm
    #     self.ui.FitResultsVertical.setPlainText("")
    #     self.ui.FitResultsVertical.append("Units: mm")
    #     self.ui.FitResultsVertical.append("Peak position %.3f" % axis1_pos)
    #     self.ui.FitResultsVertical.append("Beam width %.3f" % axis1_omega)
    #     #self.ui.FitResultsVertical.append("Height above bgr %.2f" % fitted_params_vert["I_zero"])


    #     # In principle this can function as a powermeter, but it
    #     # needs to be calibrated first
    #     total_power = np.sum(data) 
    #     self.ui.PowerMeter.display(total_power*10e-9)

        
        
    #     # Showing the beam profile itself 
    #     canvas_forbeam = self.ui.BeamDisplay.canvas
    #     canvas_forbeam.ax.pcolorfast(data)
    #     canvas_forbeam.draw()

    #     # Showing the horizontal axis fit (needs to be improved)
    #     canvas_forHorizFit = self.ui.xplot.canvas
    #     canvas_forHorizFit.ax.plot(fit0[0],fit0[1],"b-",fit0[0],mm.residual_G1D(fit0[2].params,fit0[0]),"r-")
    #     canvas_forHorizFit.ax.xaxis.set_visible(False) #for now remove axis labeling, it's wrong anyway
    #     canvas_forHorizFit.ax.yaxis.set_visible(False)
    #     canvas_forHorizFit.draw()

    #     # Showing the vertical axis fit (needs to be improved)
    #     canvas_forVertFit = self.ui.yplot.canvas
    #     canvas_forVertFit.ax.plot(fit1[1],fit1[0],"b-",mm.residual_G1D(fit1[2].params,fit1[0]),fit1[0],"r-")
    #     canvas_forVertFit.ax.xaxis.set_visible(False) #for now remove axis labeling, it's wrong anyway
    #     canvas_forVertFit.ax.yaxis.set_visible(False)
    #     canvas_forVertFit.draw()


    def plotting(self,fitresults_tuple):
        
        fit0 = fitresults_tuple[0]
        fit1 = fitresults_tuple[1]
        data = fitresults_tuple[2]
        scale_pixels_mm = 1

        if (fit0 == 1) or (fit1 == 1):
            self.ui.FitResultsHorizontal.setPlainText("")
            self.ui.FitResultsVertical.setPlainText("")
            self.ui.ErrorMessages.setPlainText("Fitting failed!")
            return None
        else:
            # Read out and display fit parameters along horiz. axis
            fitted_params_horiz = fit0[2].params.valuesdict()
            axis0_pos = fitted_params_horiz["r_zero"]*scale_pixels_mm
            axis0_omega = fitted_params_horiz["omega_zero"]*scale_pixels_mm
            self.ui.FitResultsHorizontal.setPlainText("")
            self.ui.FitResultsHorizontal.append("Units: mm")
            self.ui.FitResultsHorizontal.append("Peak position %.3f" % axis0_pos)
            self.ui.FitResultsHorizontal.append("Beam width %.3f" % axis0_omega)
            #self.ui.FitResultsHorizontal.append("Height above bgr %.3f" % fitted_params_horiz["I_zero"])

            # Read out and display fit parameters along horiz. axis
            fitted_params_vert = fit1[2].params.valuesdict()
            axis1_pos = fitted_params_vert["r_zero"]*scale_pixels_mm
            axis1_omega = fitted_params_vert["omega_zero"]*scale_pixels_mm
            self.ui.FitResultsVertical.setPlainText("")
            self.ui.FitResultsVertical.append("Units: mm")
            self.ui.FitResultsVertical.append("Peak position %.3f" % axis1_pos)
            self.ui.FitResultsVertical.append("Beam width %.3f" % axis1_omega)
            #self.ui.FitResultsVertical.append("Height above bgr %.2f" % fitted_params_vert["I_zero"])


            # In principle this can function as a powermeter, but it
            # needs to be calibrated first
            
            #total_power = np.sum(data) 
            #self.ui.PowerMeter.display(total_power*10e-9)

            
            
            # Showing the beam profile itself 
            canvas_forbeam = self.ui.BeamDisplay.canvas
            canvas_forbeam.ax.pcolorfast(data)
            canvas_forbeam.draw()

            # Showing the horizontal axis fit (needs to be improved)
            canvas_forHorizFit = self.ui.xplot.canvas
            canvas_forHorizFit.ax.plot(fit0[0],fit0[1],"b-",fit0[0],mm.residual_G1D(fit0[2].params,fit0[0]),"r-")
            canvas_forHorizFit.ax.xaxis.set_visible(False) #for now remove axis labeling, it's wrong anyway
            canvas_forHorizFit.ax.yaxis.set_visible(False)
            canvas_forHorizFit.draw()

            # Showing the vertical axis fit (needs to be improved)
            canvas_forVertFit = self.ui.yplot.canvas
            canvas_forVertFit.ax.plot(fit1[1],fit1[0],"b-",mm.residual_G1D(fit1[2].params,fit1[0]),fit1[0],"r-")
            canvas_forVertFit.ax.xaxis.set_visible(False) #for now remove axis labeling, it's wrong anyway
            canvas_forVertFit.ax.yaxis.set_visible(False)
            canvas_forVertFit.draw()


    def doFitting_file(self):
        filename = self.ui.fileOutput.toPlainText()
        
        try: 
            data = frt.make_numpyarray(filename)
            self.ui.ErrorMessages.setPlainText("")
            
            fittingThread = FitterThread(data)
            self.connect(fittingThread,QtCore.SIGNAL("plotter(PyQt_PyObject)"),self.plotting)
            fittingThread.start()

            #print(fittingThread.currentThreadId)




            #self.fitting(data)
        except:
            self.ui.FitResultsHorizontal.setPlainText("")
            self.ui.FitResultsVertical.setPlainText("")
            self.ui.ErrorMessages.setPlainText("Function doFitting_file() failed \n It's probably impossible to fit from this file or no file has been chosen")
            return None

    
    def doFitting_stream(self):
        #self.timer.start(2000)
        self.ui.ErrorMessages.setPlainText("")
        fittingThread = FitterThreadStream()
        self.connect(fittingThread,QtCore.SIGNAL("plotter(PyQt_PyObject)"),self.plotting)
        fittingThread.start()
        

    def showFileInput(self):

        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', 
             '/home/Oleksiy/Desktop/PythonCode')
        self.ui.fileOutput.setText(fname)


    def keyPressEvent(self, e):
        
        if e.key() == QtCore.Qt.Key_Escape:
            self.close()


class FitterThread(QtCore.QThread):
    """docstring for FitterThread"QThread     def __init__(self, arg):
    """

    def __init__(self,data,parent=None):
        QtCore.QThread.__init__(self,parent)
        self.data = data

    def __del__(self):
        self.wait()
        

    def fitter(self,data):
        """
        This function does the fitting along the 2 axes independently (using integrated  
        images along axes), and it can be extended to do the full 2D fitting.

        The "data" input is the 2D numpy array representing the image. The idea is that 
        this can be anything, most importantly the data coming from an Ethernet cam

        """
        print("Doing fitting")
        #print(data.shape)
        scale_pixels_mm = 1#self.ui.pixelSizeInput.value()*10e-3
    
        # Do the fits along each axis. Try if it works and show an error message if it fails
        try:
            fit0 = frt.fit_axis(self.data,1) # NOTE! Axes and array indexing are messed up
            fit1 = frt.fit_axis(self.data,0)
        except:
            return (1,1,1)

        return (fit0,fit1,self.data)

    def run(self):
        result = self.fitter(self.data)
        self.emit(QtCore.SIGNAL("plotter(PyQt_PyObject)"),result)

class FitterThreadStream(QtCore.QThread):
    """docstring for FitterThread"QThread     def __init__(self, arg):
    """

    def __init__(self,parent=None):
        QtCore.QThread.__init__(self,parent)

    def __del__(self):
        self.wait()
        

    def fitter(self,data):
        """
        This function does the fitting along the 2 axes independently (using integrated  
        images along axes), and it can be extended to do the full 2D fitting.

        The "data" input is the 2D numpy array representing the image. The idea is that 
        this can be anything, most importantly the data coming from an Ethernet cam

        """
        print("Doing fitting stream")
        #print(data.shape)
        scale_pixels_mm = 1#self.ui.pixelSizeInput.value()*10e-3
    
        # Do the fits along each axis. Try if it works and show an error message if it fails
        try:
            #print("Entering the thread loop")
            fit0 = frt.fit_axis(data,1) # NOTE! Axes and array indexing are messed up
            fit1 = frt.fit_axis(data,0)
            
        except:
            return (1,1,1)

        return (fit0,fit1,data)

    def run(self):

        data = imggrab.get_image(0)[1]
        result = self.fitter(data)
        self.emit(QtCore.SIGNAL("plotter(PyQt_PyObject)"),result)
        
# class GenericThread(QtCore.QThread):
#     def __init__(self, function, *args, **kwargs):
#             QtCore.QThread.__init__(self)
#             self.function = function
#             self.args = args
#             self.kwargs = kwargs
 
#     def __del__(self):
#             self.wait()
 
#     def run(self):
#             print("thread is running")
#             self.function(*self.args,**self.kwargs)
#             return
 
app = QtGui.QApplication(sys.argv)
myWindow = MyWindowClass()
#myWindow.showFullScreen()
myWindow.show()
sys.exit(app.exec_())


if __name__ == '__main__':

    #sys.path.append("/Users/oleksiy/Desktop/PythonCode/beamFitter/GUIdesign")
    main()
    sys.exit(0)

