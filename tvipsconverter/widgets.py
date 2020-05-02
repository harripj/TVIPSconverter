from PyQt5 import uic
from PyQt5.QtWidgets import (QApplication, QFileDialog, QGraphicsScene,
                             QGraphicsPixmapItem, QMainWindow, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal
from utils import recorder as rec
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as
                                                FigureCanvas)
from pathlib import Path
import logging
from time import sleep
logging.basicConfig(level=logging.DEBUG)
sys.path.append(".")

# import the UI interface
rawgui, Window = uic.loadUiType("./tvipsconverter/widget_2.ui")

#
# class export_to_hdf5_thread(QThread):
#     readfile = pyqtSignal(int)
#
#     def __init__(self, ipath, opath, improc, vbfsett, progbar):
#         QThread.__init__(self)
#         self.inpath = ipath
#         self.oupath = opath
#         self.improc = improc
#         self.vbfsettings = vbfsett
#         self.progressBar = progbar
#
#     def __del__(self):
#         self.wait()
#
#     def run(self):
#         rec.convertToHDF5(self.inpath, self.oupath, self.improc,
#                           self.vbfsettings, progbar=self.progressBar)


class External(QThread):
    """
    Runs a counter thread.
    """
    countChanged = pyqtSignal(int)
    finish = pyqtSignal()

    def __init__(self, fin):
        QThread.__init__(self)
        self.fin = fin

    def run(self):
        count = 0
        while count < self.fin:
            count += 1
            sleep(0.1)
            self.countChanged.emit(count)
        self.finish.emit()


class ConnectedWidget(rawgui):
    """Class connecting the gui elements to the back-end functionality"""
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.setupUi(window)

        self.original_preview = None
        self.path_preview = None

        self.connectUI()

    def connectUI(self):
        # initial browse button
        self.pushButton.clicked.connect(self.open_tvips_file)

        # update button on the preview
        self.pushButton_4.clicked.connect(self.updatePreview)
        #
        # # execute the conversion command
        self.pushButton_3.clicked.connect(self.get_hdf5_path)
        # shitty workaround for saving to hdf5 with updated gui
        self.pushButton_6.clicked.connect(self.write_to_hdf5)
        # self.pushButton_3.clicked.connect(self.exportFiles)
        # test threading
        self.pushButton_2.clicked.connect(self.threadCheck)
        #
        # # deactivate part of the gui upon activation
        # self.checkBox_8.stateChanged.connect(self.updateActive)
        # self.checkBox_3.stateChanged.connect(self.updateActive)
        # self.checkBox_7.stateChanged.connect(self.updateActive)
        # self.checkBox_5.stateChanged.connect(self.updateActive)
        # self.checkBox_4.stateChanged.connect(self.updateActive)
        # self.checkBox_6.stateChanged.connect(self.updateActive)
        # self.updateActive()

    def threadCheck(self):
        self.calc = External(50)
        self.calc.countChanged.connect(self.onCountChanged)
        self.calc.finish.connect(self.done)
        self.calc.start()
        self.window.setEnabled(False)

    def onCountChanged(self, value):
        self.progressBar.setValue(value)

    def done(self):
        self.window.setEnabled(True)

    def open_tvips_file(self):
        path = self.openFileBrowser("TVIPS (*.tvips)")
        # check if it's a valid file
        if path:
            try:
                rec.Recorder.valid_first_tvips_file(path)
                self.lineEdit.setText(path)
                self.statusedit.setText("Selected tvips file")
            except Exception as e:
                self.lineEdit.setText("")
                self.statusedit.setText(str(e))

    def openFileBrowser(self, fs):
        path, okpres = QFileDialog.getOpenFileName(caption="Select file",
                                                   filter=fs)
        if okpres:
            return str(Path(path))

    def saveFileBrowser(self, fs):
        path, okpres = QFileDialog.getSaveFileName(caption="Select file",
                                                   filter=fs)
        if okpres:
            return str(Path(path))

    def openFolderBrowser(self):
        path = QFileDialog.getExistingDirectory(caption="Choose directory")
        if path:
            return str(Path(path))

    def read_modsettings(self):
        path = self.lineEdit.text()
        improc = {
            "useint": self.checkBox_8.checkState(),
            "whichint": self.spinBox_11.value(),
            "usebin": self.checkBox_5.checkState(),
            "whichbin": self.spinBox_3.value(),
            "usegaus": self.checkBox_3.checkState(),
            "gausks": self.spinBox_6.value(),
            "gaussig": self.doubleSpinBox.value(),
            "usemed": self.checkBox_7.checkState(),
            "medks": self.spinBox_10.value(),
            "usels": self.checkBox_4.checkState(),
            "lsmin": self.spinBox_7.value(),
            "lsmax": self.spinBox_8.value(),
            "usecoffset": self.checkBox.checkState()
        }
        vbfsettings = {
            "calcvbf": self.checkBox_10.checkState(),
            "vbfrad": self.spinBox_12.value(),
            "vbfxoffset": self.spinBox_13.value(),
            "vbfyoffset": self.spinBox_14.value()
        }
        return path, improc, vbfsettings

    def updatePreview(self):
        """Read the first image from the file and create a preview"""
        # read the gui info
        path, improc, vbfsettings = self.read_modsettings()
        # create one image properly processed
        try:
            if not path:
                raise Exception("A TVIPS file must be selected!")
            # get and calculate the image. Also get old image and new image
            # size. only change original image if there is none or the path
            # has changed
            if (self.original_preview is None) or (self.path_preview != path):
                self.update_line(self.statusedit, "Extracting frame...")
                self.original_preview = rec.getOriginalPreviewImage(
                                                path, improc=improc,
                                                vbfsettings=vbfsettings)
            # update the path
            self.path_preview = path
            ois = self.original_preview.shape
            filterframe = rec.filter_image(self.original_preview, **improc)
            nis = filterframe.shape
            # check if the VBF aperture fits in the frame
            if vbfsettings["calcvbf"]:
                midx = nis[1]//2
                midy = nis[0]//2
                xx = vbfsettings["vbfxoffset"]
                yy = vbfsettings["vbfyoffset"]
                rr = vbfsettings["vbfrad"]
                if (midx+xx-rr < 0 or
                   midx+xx+rr > nis[1] or
                   midy+yy-rr < 0 or
                   midy+yy-rr > nis[0]):
                    raise Exception("Virtual bright field aperture out "
                                    "of bounds")
            # plot the image and the circle over it
            fig = plt.figure(frameon=False,
                             figsize=(filterframe.shape[1]/100,
                                      filterframe.shape[0]/100))
            canvas = FigureCanvas(fig)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(filterframe, cmap="Greys_r")
            if vbfsettings["calcvbf"]:
                xoff = vbfsettings["vbfxoffset"]
                yoff = vbfsettings["vbfyoffset"]
                circ = Circle((filterframe.shape[1]//2+xoff,
                               filterframe.shape[0]//2+yoff),
                              vbfsettings["vbfrad"],
                              color="red",
                              alpha=0.5)
                ax.add_patch(circ)
            canvas.draw()
            scene = QGraphicsScene()
            scene.addWidget(canvas)
            self.graphicsView.setScene(scene)
            self.graphicsView.fitInView(scene.sceneRect())
            self.repaint_widget(self.graphicsView)
            self.update_line(self.statusedit, "Succesfully created preview.")
            self.update_line(self.lineEdit_8, f"Original: {ois[0]}x{ois[1]}. "
                                              f"New: {nis[0]}x{nis[1]}.")
            plt.close(fig)
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")
            # empty the preview
            self.update_line(self.lineEdit_8, "")
            scene = QGraphicsScene()
            self.graphicsView.setScene(scene)
            self.repaint_widget(self.graphicsView)
            self.original_preview = None
            self.path_preview = None

    def repaint_widget(self, widget):
        widget.hide()
        widget.show()

    def update_line(self, line, string):
        line.setText(string)
        line.hide()
        line.show()

    def get_hdf5_path(self):
        # open a savefile browser
        try:
            # read the gui info
            (self.inpath, self.improc,
             self.vbfsettings) = self.read_modsettings()
            if not self.inpath:
                raise Exception("A TVIPS file must be selected!")
            self.oupath = self.saveFileBrowser("HDF5 (*.hdf5)")
            if not self.oupath:
                raise Exception("No valid HDF5 file path selected")
            self.lineEdit_2.setText(self.oupath)
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")

    def write_to_hdf5(self):
        # try:
        (self.inpath, self.improc,
         self.vbfsettings) = self.read_modsettings()
        if not self.inpath:
            raise Exception("A TVIPS file must be selected!")
        self.oupath = self.lineEdit_2.text()
        if not self.oupath:
            raise Exception("No valid HDF5 file path selected")
        # read the gui info
        self.update_line(self.statusedit, f"Exporting to {self.oupath}")
        path = self.inpath
        opath = self.oupath
        improc = self.improc
        vbfsettings = self.vbfsettings
        self.get_thread = rec.Recorder(path,
                                       improc=improc,
                                       vbfsettings=vbfsettings,
                                       outputpath=opath)
        self.get_thread.increase_progress.connect(self.increase_progbar)
        self.get_thread.finish.connect(self.done)
        self.get_thread.start()
        self.window.setEnabled(False)
        self.update_line(self.statusedit,
                         f"Succesfully exported to HDF5")
        # except Exception as e:
        #    self.update_line(self.statusedit, f"Error: {e}")

    def increase_progbar(self, value):
        self.progressBar.setValue(value)

    def hardRepaint(self):
        self.window.hide()
        self.window.show()


def main():
    app = QApplication([])
    window = Window()
    form = ConnectedWidget(window)
    window.setWindowTitle("TVIPS converter")
    window.show()
    form.lineEdit.setText("/Volumes/Elements/200309-2F/"
                          "rec_20200309_113857_000.tvips")
    form.lineEdit_2.setText("/Users/nielscautaerts/Desktop/stream_2.hdf5")
    app.exec_()


def mainDebug():
    app = QApplication([])
    window = Window()
    form = ConnectedWidget(window)
    window.setWindowTitle("TVIPS / blo converter")
    window.show()
    # set the values
    form.lineEdit.setText("./Dummy/rec_20190412_183600_000.tvips")
    form.spinBox.setValue(150)
    form.spinBox_2.setValue(1)
    form.lineEdit_2.setText("./Dummy")
    form.lineEdit_3.setText("testpref")
    form.comboBox.setCurrentIndex(1)
    app.exec_()


if __name__ == "__main__":
    main()
