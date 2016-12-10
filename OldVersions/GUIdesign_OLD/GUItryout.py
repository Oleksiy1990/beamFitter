import sys
from PyQt4 import QtGui, QtCore
from mainwindow import Ui_MainWindow

sys.path.append("C:\\Users\\Oleksiy\\Desktop\\Code\\beamFitter")

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import file1
from lmfit import Parameters, minimize, fit_report
import numpy as np
from scipy import ndimage


class MyWindowClass(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        QtCore.QObject.connect(self.ui.StartButton, QtCore.SIGNAL('clicked()'), self.doFitting)
        QtCore.QObject.connect(self.ui.fileDialog, QtCore.SIGNAL('clicked()'), self.showFileInput)
        
    

    def doFitting(self):
        #t = np.arange(1.0, 5.0, 0.01)
        #s = np.sin(2*np.pi*t)
        #self.ui.xplot.canvas.ax.plot(t, s)
        #self.ui.xplot.canvas.draw()

        filename = self.ui.fileOutput.toPlainText()
        self.ui.FitResultsSummary.setPlainText("These are fitting results")
        self.ui.ErrorMessages.setPlainText("")
        try: 
            data = file1.make_numpyarray(filename)
        except:
            self.ui.ErrorMessages.setPlainText("Something went wrong with fitting")
            return None
        fit1 = file1.fit_axis(data,0)
        report = fit_report(fit1[2])
        self.ui.FitResultsSummary.append(report)
        
        # rotating by 90 degrees on the other axis doesn't work well yet
        #t = np.arange(1.0, 5.0, 0.01)
        #s = np.sin(2*np.pi*t)
        #self.myplot = self.ui.yplot.canvas.ax.plot(t, s)
        #self.rotated = ndimage.rotate(self.myplot,90)
        #self.rotated.draw()

        self.ui.BeamDisplay.canvas.ax.pcolorfast(data)
        self.ui.BeamDisplay.canvas.draw()

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

    sys.path.append("C:\\Users\\Oleksiy\\Desktop\\Code\\beamFitter\\GUIdesign")
    main()
    sys.exit(0)

