import sys
from PyQt4 import QtGui, QtCore
from mainwindow import Ui_MainWindow

sys.path.append('C:\\Users\\Oleksiy\\Desktop\\Code\\beamFitter')

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

from lmfit import Parameters, minimize, fit_report
import numpy as np



import imageUSB
import fitterroutines as frt
import mathmodels as mm 



class MyWindowClass(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        QtCore.QObject.connect(self.ui.StartButton, QtCore.SIGNAL('clicked()'), self.doFitting_stream)
        QtCore.QObject.connect(self.ui.fitFromFile, QtCore.SIGNAL('clicked()'), self.doFitting_file)
        QtCore.QObject.connect(self.ui.fileDialog, QtCore.SIGNAL('clicked()'), self.showFileInput)
        QtCore.QObject.connect(self.ui.PauseButton, QtCore.SIGNAL('clicked()'), self.close)

    
    def fitting(self,data):
        
        print("Doing fitting")
        print(data.shape)
        scale_pixels_mm = self.ui.pixelSizeInput.value()*10e-3
    

        fit0 = frt.fit_axis(data,1) # NOTE! Axes and array indexing are messed up
        fit1 = frt.fit_axis(data,0)
        #report0 = fit_report(fit0[2])
        
        fitted_params_horiz = fit0[2].params.valuesdict()
        axis0_pos = fitted_params_horiz["r_zero"]*scale_pixels_mm
        axis0_omega = fitted_params_horiz["omega_zero"]*scale_pixels_mm
        self.ui.FitResultsHorizontal.setPlainText("")
        self.ui.FitResultsHorizontal.append("Units: mm")
        self.ui.FitResultsHorizontal.append("Peak position %.3f" % axis0_pos)
        self.ui.FitResultsHorizontal.append("Beam width %.3f" % axis0_omega)
        #self.ui.FitResultsHorizontal.append("Height above bgr %.3f" % fitted_params_horiz["I_zero"])

        fitted_params_vert = fit1[2].params.valuesdict()
        axis1_pos = fitted_params_vert["r_zero"]*scale_pixels_mm
        axis1_omega = fitted_params_vert["omega_zero"]*scale_pixels_mm
        self.ui.FitResultsVertical.setPlainText("")
        self.ui.FitResultsVertical.append("Units: mm")
        self.ui.FitResultsVertical.append("Peak position %.3f" % axis1_pos)
        self.ui.FitResultsVertical.append("Beam width %.3f" % axis1_omega)
        #self.ui.FitResultsVertical.append("Height above bgr %.2f" % fitted_params_vert["I_zero"])


        total_power = np.sum(data)
        self.ui.PowerMeter.display(total_power*10e-9)

        
        
        # Showing the beam profile itself 
        canvas_forbeam = self.ui.BeamDisplay.canvas
        canvas_forbeam.ax.pcolorfast(data)
        canvas_forbeam.draw()

        # Showing the horizontal axis fit (needs to be improved)
        canvas_forHorizFit = self.ui.xplot.canvas
        canvas_forHorizFit.ax.plot(fit0[0],fit0[1],"b-",fit0[0],mm.residual_G1D(fit0[2].params,fit0[0]),"r-")
        canvas_forHorizFit.draw()

        # Showing the vertical axis fit (needs to be improved)
        canvas_forVertFit = self.ui.yplot.canvas
        canvas_forVertFit.ax.plot(fit1[1],fit1[0],"b-",mm.residual_G1D(fit1[2].params,fit1[0]),fit1[0],"r-")
        canvas_forVertFit.draw()

    def doFitting_file(self):
        filename = self.ui.fileOutput.toPlainText()
        
        try: 
            data = frt.make_numpyarray(filename)
            self.ui.ErrorMessages.setPlainText("")
            self.fitting(data)
        except:
            self.ui.FitResultsHorizontal.setPlainText("")
            self.ui.FitResultsVertical.setPlainText("")
            self.ui.ErrorMessages.setPlainText("Something went wrong with fitting")
            return None

    def doFitting_stream(self):
        while True:
            data = imageUSB.get_image(0)
            #print(type(data[1]))
            #print(data[1].shape)
            #break
            if data[0]:
                print(data[0])
                self.fitting(data[1])
                break #somehow for now it cannot run normally in a loop, it doesn't display images and crashes
            else:
                self.ui.ErrorMessages.setPlainText("Cannot get data from camera") 
       
        

        

    def showFileInput(self):

        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', 
             '/home/Oleksiy/Desktop/PythonCode')
        self.ui.fileOutput.setText(fname)


    def keyPressEvent(self, e):
        
        if e.key() == QtCore.Qt.Key_Escape:
            self.close()

 
app = QtGui.QApplication(sys.argv)
myWindow = MyWindowClass()
myWindow.showFullScreen()
sys.exit(app.exec_())


if __name__ == '__main__':

    sys.path.append("/Users/oleksiy/Desktop/PythonCode/beamFitter/GUIdesign")
    main()
    sys.exit(0)

