# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(693, 554)
        self.centralWidget = QtGui.QWidget(MainWindow)
        self.centralWidget.setObjectName(_fromUtf8("centralWidget"))
        self.BeamDisplay = MplWidget(self.centralWidget)
        self.BeamDisplay.setGeometry(QtCore.QRect(30, 280, 251, 191))
        self.BeamDisplay.setObjectName(_fromUtf8("BeamDisplay"))
        self.xplot = MplWidget(self.centralWidget)
        self.xplot.setGeometry(QtCore.QRect(30, 150, 256, 100))
        self.xplot.setObjectName(_fromUtf8("xplot"))
        self.yplot = MplWidget(self.centralWidget)
        self.yplot.setGeometry(QtCore.QRect(300, 210, 100, 256))
        self.yplot.setObjectName(_fromUtf8("yplot"))
        self.TwoD_region = QtGui.QGraphicsView(self.centralWidget)
        self.TwoD_region.setGeometry(QtCore.QRect(590, 10, 91, 91))
        self.TwoD_region.setObjectName(_fromUtf8("TwoD_region"))
        self.TwoD_fit = QtGui.QGraphicsView(self.centralWidget)
        self.TwoD_fit.setGeometry(QtCore.QRect(590, 110, 91, 81))
        self.TwoD_fit.setObjectName(_fromUtf8("TwoD_fit"))
        self.FitResultsSummary = QtGui.QTextBrowser(self.centralWidget)
        self.FitResultsSummary.setGeometry(QtCore.QRect(420, 180, 161, 121))
        self.FitResultsSummary.setObjectName(_fromUtf8("FitResultsSummary"))
        self.ErrorMessages = QtGui.QTextBrowser(self.centralWidget)
        self.ErrorMessages.setGeometry(QtCore.QRect(420, 340, 161, 131))
        self.ErrorMessages.setObjectName(_fromUtf8("ErrorMessages"))
        self.label = QtGui.QLabel(self.centralWidget)
        self.label.setGeometry(QtCore.QRect(320, 20, 331, 31))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Courier New"))
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.StartButton = QtGui.QPushButton(self.centralWidget)
        self.StartButton.setGeometry(QtCore.QRect(30, 20, 115, 32))
        self.StartButton.setObjectName(_fromUtf8("StartButton"))
        self.PauseButton = QtGui.QPushButton(self.centralWidget)
        self.PauseButton.setGeometry(QtCore.QRect(160, 20, 115, 32))
        self.PauseButton.setObjectName(_fromUtf8("PauseButton"))
        self.fileDialog = QtGui.QPushButton(self.centralWidget)
        self.fileDialog.setGeometry(QtCore.QRect(410, 60, 115, 32))
        self.fileDialog.setObjectName(_fromUtf8("fileDialog"))
        self.fileOutput = QtGui.QTextBrowser(self.centralWidget)
        self.fileOutput.setGeometry(QtCore.QRect(40, 70, 361, 21))
        self.fileOutput.setObjectName(_fromUtf8("fileOutput"))
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtGui.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 693, 22))
        self.menuBar.setMouseTracking(False)
        self.menuBar.setObjectName(_fromUtf8("menuBar"))
        self.menuSome_menu = QtGui.QMenu(self.menuBar)
        self.menuSome_menu.setMouseTracking(False)
        self.menuSome_menu.setObjectName(_fromUtf8("menuSome_menu"))
        self.menuSome_other_menu = QtGui.QMenu(self.menuBar)
        self.menuSome_other_menu.setObjectName(_fromUtf8("menuSome_other_menu"))
        MainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QtGui.QToolBar(MainWindow)
        self.mainToolBar.setObjectName(_fromUtf8("mainToolBar"))
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtGui.QStatusBar(MainWindow)
        self.statusBar.setObjectName(_fromUtf8("statusBar"))
        MainWindow.setStatusBar(self.statusBar)
        self.menuBar.addAction(self.menuSome_menu.menuAction())
        self.menuBar.addAction(self.menuSome_other_menu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.FitResultsSummary.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'.Helvetica Neue DeskInterface\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'MS Shell Dlg 2\'; font-size:8.25pt;\"><br /></p></body></html>", None))
        self.label.setText(_translate("MainWindow", "Python gaussian beam fitter", None))
        self.StartButton.setText(_translate("MainWindow", "Start", None))
        self.PauseButton.setText(_translate("MainWindow", "Pause", None))
        self.fileDialog.setText(_translate("MainWindow", "Choose file", None))
        self.menuSome_menu.setTitle(_translate("MainWindow", "some menu", None))
        self.menuSome_other_menu.setTitle(_translate("MainWindow", "some other menu", None))

from mplwidget import MplWidget
