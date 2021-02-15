from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QFileDialog, QGraphicsScene
from PyQt5.QtCore import QByteArray, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QImage, QPixmap
import pyqtgraph as pg
import sys

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pathlib import Path
import logging
from time import sleep
import numpy as np
import os

# hotfix 3.9 MacOS Big Sur bug
if sys.platform == "darwin":
    os.environ["QT_MAC_WANTS_LAYER"] = "1"

from .utils import recorder as rec
from .utils import blockfile as blf
from .utils import tiffexport as tfe
from .utils import hspy as hspf

logging.basicConfig(level=logging.DEBUG)
sys.path.append(".")

logger = logging.getLogger(__name__)
# import the UI interface
rawgui, Window = uic.loadUiType(
    str(Path(__file__).parent.absolute()) + os.sep + "widget.ui"
)


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

        # has a valid hdf5 file been selected or not
        self.valid_hdf5 = False

        # data storage for preview (only need figure)
        self.fig_prev = None

        # data storage for vbf preview
        self.vbf_data = None
        self.vbf_sets = None  # settings
        self.fig_vbf = None
        self.vbf_im = None

        # init attributes for live view diffraction
        self.fig_diffraction = None
        self.ai_diffraction = None
        self.mouse_in_axes = False  # interactive flags
        self.mouse_pressed_vbf = False

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
        self.lineEdit_4.textChanged.connect(self.auto_read_hdf5)
        # self.pushButton_3.clicked.connect(self.exportFiles)
        # exporting the preview
        self.pushButton_2.clicked.connect(self.export_preview)
        # browse to select a file
        self.pushButton_8.clicked.connect(self.select_hdf5_file)
        # auto update custom scan dimensions (spinbox 16)
        self.spinBox.valueChanged.connect(self.update_final_frame)
        self.spinBox_2.valueChanged.connect(self.update_final_frame)
        self.spinBox_15.valueChanged.connect(self.update_final_frame)
        self.checkBox_2.stateChanged.connect(self.update_final_frame)
        # create a preview of the vbf
        self.pushButton_10.clicked.connect(self.update_vbf)
        self.spinBox.valueChanged.connect(self.auto_update_vbf)
        self.spinBox_2.valueChanged.connect(self.auto_update_vbf)
        self.spinBox_15.valueChanged.connect(self.auto_update_vbf)
        self.spinBox_9.valueChanged.connect(self.auto_update_vbf)
        self.comboBox_hyst_dir.currentTextChanged.connect(self.auto_update_vbf)
        self.comboBox_snakescan_dir.currentTextChanged.connect(self.auto_update_vbf)
        self.checkBox_6.stateChanged.connect(self.auto_update_vbf)
        self.checkBox_12.stateChanged.connect(self.auto_update_vbf)

        # connecting the horizontal sliders
        self.horizontalSlider.sliderReleased.connect(self.update_levels_vbf)
        self.horizontalSlider_2.sliderReleased.connect(self.update_levels_vbf)
        # saving the VBF
        self.pushButton_7.clicked.connect(self.export_vbf)
        # # deactivate part of the gui upon activation
        # browsing blo file
        self.pushButton_9.clicked.connect(self.get_blo_path)
        # start blo conversion
        self.pushButton_5.clicked.connect(self.write_to_file)
        # select tiff file
        self.pushButton_12.clicked.connect(self.get_tiff_path)
        # export tiff files
        self.pushButton_11.clicked.connect(self.export_tiffs)
        # show cropped region
        self.pushButton_13.clicked.connect(self.show_cropped_region)
        # write scan settings (from VBF preview) to hdf5
        # self.pushButton_write_scan_parameters.clicked.connect(
        #     self.write_scan_parameters_hdf5
        # )

    def export_preview(self):
        try:
            if self.fig_prev is None:
                raise Exception("Must first create preview")
            path = self.saveFileBrowser("PNG (*.png)")
            if path is None:
                raise Exception("No valid file selected")
            self.fig_prev.savefig(path)
            self.update_line(self.statusedit, "Succesfully saved preview.")
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")

    def export_vbf(self):
        try:
            if self.fig_vbf is None:
                raise Exception("Must first create VBF preview")
            path = self.saveFileBrowser("PNG (*.png)")
            if path is None:
                raise Exception("No valid file selected")
            self.fig_vbf.savefig(path)
            self.update_line(self.statusedit, "Succesfully saved VBF.")
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")

    def update_levels_vbf(self):
        if self.vbf_data is not None:
            vmin = self.horizontalSlider.value()
            vmax = self.horizontalSlider_2.value()
            mn = self.vbf_data.min()
            mx = self.vbf_data.max()
            unit = (mx - mn) / 100
            climmin = mn + vmin * unit
            climmax = mn + (vmax + 1) * unit

            self.vbf_im.set_clim(climmin, climmax)
            # self.fig_vbf.axes[0].update_artist(self.vbf_im)
            self.fig_vbf.canvas.blit(self.fig_vbf.axes[0].bbox)

            logger.debug("updated levels vbf")

    def save_vbf_to_hdf5(self):
        pass

    def auto_update_vbf(self):
        if self.checkBox_auto_update_vbf.isChecked():
            self.update_vbf()

    def update_final_frame(self):
        # update x, y crop box values
        self.spinBox_22.setValue(0)
        self.spinBox_20.setValue(0)
        self.spinBox_23.setValue(self.spinBox.value())
        self.spinBox_21.setValue(self.spinBox_2.value())

        if self.checkBox_2.checkState():
            # we use self defined size
            start = self.spinBox_15.value()
            frms = self.spinBox.value() * self.spinBox_2.value()
            self.spinBox_16.setValue(start + frms - 1)
        else:
            # we use auto-size
            start = self.spinBox_15.value()
            frms = self.lineEdit_11.text()
            try:
                dim = np.sqrt(int(frms))
                self.spinBox_16.setValue(start + dim ** 2 - 1)
            except Exception:
                self.spinBox_16.setValue(0)

    def threadCheck(self):
        self.calc = External(50)
        self.calc.countChanged.connect(self.onCountChanged)
        self.calc.finish.connect(self.done)
        self.calc.start()
        self.window.setEnabled(False)

    def onCountChanged(self, value):
        self.progressBar.setValue(value)

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
        path, okpres = QFileDialog.getOpenFileName(caption="Select file", filter=fs)
        if okpres:
            return str(Path(path))

    def saveFileBrowser(self, fs):
        path, okpres = QFileDialog.getSaveFileName(caption="Select file", filter=fs)
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
            "usecoffset": self.checkBox.checkState(),
            "bintype": self.radioButton_decimation.isChecked(),  # if True then use decimation otherwise box averaging
        }
        vbfsettings = {
            "calcvbf": self.checkBox_10.checkState(),
            "vbfrad": self.spinBox_12.value(),
            "vbfxoffset": self.spinBox_13.value(),
            "vbfyoffset": self.spinBox_14.value(),
        }
        return path, improc, vbfsettings

    def updatePreview(self):
        """Read the first image from the file and create a preview"""
        # read the gui info
        path, improc, vbfsettings = self.read_modsettings()
        framenum = self.spinBox_17.value()
        # create one image properly processed
        try:
            if not path:
                raise Exception("A TVIPS file must be selected!")
            # get and calculate the image. Also get old image and new image
            # size. only change original image if there is none or the path
            # has changed
            # if (self.original_preview is None) or
            # (self.path_preview != path):
            self.update_line(self.statusedit, "Extracting frame...")
            self.original_preview = rec.getOriginalPreviewImage(
                path, improc=improc, vbfsettings=vbfsettings, frame=framenum
            )
            # update the path
            self.path_preview = path
            ois = self.original_preview.shape
            filterframe = rec.filter_image(self.original_preview, **improc)
            nis = filterframe.shape
            # check if the VBF aperture fits in the frame
            if vbfsettings["calcvbf"]:
                midx = nis[1] // 2
                midy = nis[0] // 2
                xx = vbfsettings["vbfxoffset"]
                yy = vbfsettings["vbfyoffset"]
                rr = vbfsettings["vbfrad"]
                if (
                    midx + xx - rr < 0
                    or midx + xx + rr > nis[1]
                    or midy + yy - rr < 0
                    or midy + yy - rr > nis[0]
                ):
                    raise Exception("Virtual bright field aperture out of bounds")
            # plot the image and the circle over it
            if self.fig_prev is not None:
                plt.close(self.fig_prev)
            self.fig_prev = plt.figure(
                frameon=False,
                figsize=(filterframe.shape[1] / 100, filterframe.shape[0] / 100),
            )
            canvas = FigureCanvas(self.fig_prev)
            ax = plt.Axes(self.fig_prev, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            self.fig_prev.add_axes(ax)
            ax.imshow(filterframe, cmap="Greys_r")
            if vbfsettings["calcvbf"]:
                xoff = vbfsettings["vbfxoffset"]
                yoff = vbfsettings["vbfyoffset"]
                circ = Circle(
                    (
                        filterframe.shape[1] // 2 + xoff,
                        filterframe.shape[0] // 2 + yoff,
                    ),
                    vbfsettings["vbfrad"],
                    color="red",
                    alpha=0.5,
                )
                ax.add_patch(circ)
            canvas.draw()
            scene = QGraphicsScene()
            scene.addWidget(canvas)
            self.graphicsView.setScene(scene)
            self.graphicsView.fitInView(scene.sceneRect())
            self.repaint_widget(self.graphicsView)
            self.update_line(self.statusedit, "Succesfully created preview.")
            self.update_line(
                self.lineEdit_8,
                f"Original: {ois[0]}x{ois[1]}. " f"New: {nis[0]}x{nis[1]}.",
            )
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
            (self.inpath, self.improc, self.vbfsettings) = self.read_modsettings()
            if not self.inpath:
                raise Exception("A TVIPS file must be selected!")
            self.oupath = self.saveFileBrowser("HDF5 (*.hdf5)")
            if not self.oupath:
                raise Exception("No valid HDF5 file path selected")
            self.lineEdit_2.setText(self.oupath)
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")

    def select_hdf5_file(self):
        # open an open file browser
        try:
            # read the gui info
            hdf5path = self.openFileBrowser("HDF5 (*.hdf5)")
            if not hdf5path:
                raise Exception("No valid HDF5 file path selected")
            self.lineEdit_4.setText(hdf5path)
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")

    def get_blo_path(self):
        # open a savefile browser
        try:
            filetype = self.comboBox.currentText()
            if ".blo" == filetype:
                self.oupath = self.saveFileBrowser("BLO (*.blo)")
            elif ".hspy" == filetype:
                self.oupath = self.saveFileBrowser("HSPY (*.hspy)")
            else:
                raise ValueError("Unrecognized filetype")
            if not self.oupath:
                raise Exception("No valid BLO file path selected")
            self.lineEdit_7.setText(self.oupath)
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")

    def get_tiff_path(self):
        # open a savefile browser
        try:
            self.oupath = self.saveFileBrowser("tiff (*.tiff)")
            if not self.oupath:
                raise Exception("No valid tiff file path selected")
            self.lineEdit_9.setText(self.oupath)
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")

    def done(self):
        self.window.setEnabled(True)

    @property
    def image_range(self):
        if self.checkBox_13.checkState():
            return self.spinBox_19.value(), self.spinBox_18.value()
        else:
            return None, None

    def write_to_hdf5(self):
        try:
            (self.inpath, self.improc, self.vbfsettings) = self.read_modsettings()
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
            start_frame, end_frame = self.image_range
            self.get_thread = rec.Recorder(
                path,
                improc=improc,
                vbfsettings=vbfsettings,
                outputpath=opath,
                imrange=(start_frame, end_frame),
                calcmax=self.checkBox_maxiumum_image.isChecked(),  # options kwarg
                calcave=self.checkBox_average_image.isChecked(),  # options kwarg
                calcsum=self.checkBox_sum_image.isChecked(),  # options kwarg
                refine_center=(  # options kwarg
                    self.checkBox_refine_center.isChecked(),
                    self.spinBox_refine_center_diameter.value(),
                    self.spinBox_refine_center_sigma.value(),
                ),
            )
            self.get_thread.increase_progress.connect(self.increase_progbar)
            self.get_thread.finish.connect(self.done_hdf5export)
            self.get_thread.start()
            self.window.setEnabled(False)
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")

    def done_hdf5export(self):
        self.window.setEnabled(True)
        # also update lines in the second pannel
        self.update_line(self.statusedit, "Succesfully exported to HDF5")
        # don't auto update, the gui may be before the file exists
        # self.update_line(self.lineEdit_4, self.lineEdit_2.text())

    def auto_read_hdf5(self):
        """Update HDF5 field info if lineEdit_4 (path) is changed"""
        try:
            f = rec.hdf5Intermediate(self.lineEdit_4.text())
            tot, star, en, roti, dim, imdimx, imdimy = f.get_scan_info()
            if tot is not None:
                self.update_line(self.lineEdit_3, str(tot))
            else:
                self.update_line(self.lineEdit_3, "?")
            if star is not None:
                self.update_line(self.lineEdit_5, str(star))
                logger.info(f"start: {star}")
                self.spinBox_15.setValue(star)
                self.checkBox_11.setChecked(True)
            else:
                self.update_line(self.lineEdit_5, "?")
            if en is not None:
                self.update_line(self.lineEdit_6, str(en))
            else:
                self.update_line(self.lineEdit_6, "?")
            if roti is not None:
                self.update_line(self.lineEdit_11, str(roti))
            else:
                self.update_line(self.lineEdit_11, "?")
            if dim is not None:
                if isinstance(dim, (list, tuple)):
                    self.update_line(self.lineEdit_12, str(dim))
                    self.spinBox.setValue(dim[0])
                    self.spinBox_2.setValue(dim[1])
                    self.checkBox_2.setChecked(True)
                else:
                    self.update_line(
                        self.lineEdit_12, f"{str(int(dim))}x{str(int(dim))}"
                    )
            else:
                self.update_line(self.lineEdit_12, "?")
            self.update_line(self.lineEdit_13, f"{str(imdimx)}x{str(imdimy)}")
            self.update_final_frame()
            f.close()
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")
            self.update_line(self.lineEdit_3, "")
            self.update_line(self.lineEdit_5, "")
            self.update_line(self.lineEdit_6, "")
            self.update_line(self.lineEdit_11, "")
            self.update_line(self.lineEdit_12, "")
            self.update_line(self.lineEdit_12, "")

    def update_vbf(self):
        """Calculate the virtual bf """
        path_hdf5 = self.lineEdit_4.text()
        try:
            # check if an hdf5 file is selected
            if not path_hdf5:
                raise Exception("No valid HDF5 file selected!")

            scan_exp_data = self.get_scan_export_data()
            # try to read the info from the file
            f = rec.hdf5Intermediate(path_hdf5)

            # calculate the image
            logger.debug(f"We try to create a VBF image with data: {scan_exp_data}.")
            self.vbf_data = f.get_vbf_image(**scan_exp_data)
            logger.debug("Succesfully created the VBF array")

            # save the settings for later storage
            self.vbf_sets = {
                "start_frame": scan_exp_data["start_frame"],
                "end_frame": scan_exp_data["end_frame"],
                "scan_dim_x": scan_exp_data["sdimx"],
                "scan_dim_y": scan_exp_data["sdimy"],
                "hysteresis": scan_exp_data["hyst"],
                "winding_scan": scan_exp_data["snakescan"],
            }

            # plot the image and store it for further use. First close prior
            # image
            if self.fig_vbf is None:
                self.fig_vbf = plt.Figure(
                    frameon=False,
                    figsize=(
                        self.vbf_data.shape[1] / 100,
                        self.vbf_data.shape[0] / 100,
                    ),
                )

                self.canvas_vbf = FigureCanvas(self.fig_vbf)
                ax = plt.Axes(self.fig_vbf, [0.0, 0.0, 1.0, 1.0])
                ax.set_axis_off()
                self.fig_vbf.add_axes(ax)
                self.vbf_im = ax.imshow(self.vbf_data, cmap="plasma")

                # draw crosshairs
                self.crosshair_horizontal = ax.axhline(0, color="k", alpha=0)
                self.crosshair_vertical = ax.axvline(0, color="k", alpha=0)

                # connect interactive features
                self.cid = []
                self.cid.append(
                    self.fig_vbf.canvas.mpl_connect(
                        "figure_enter_event", self.mouse_enter_axes_vbf
                    )
                )
                self.cid.append(
                    self.fig_vbf.canvas.mpl_connect(
                        "figure_leave_event", self.mouse_leave_axes_vbf
                    )
                )
                self.cid.append(
                    self.fig_vbf.canvas.mpl_connect(
                        "motion_notify_event", self.mouse_moved_update_diffraction
                    )
                )
                self.cid.append(
                    self.canvas_vbf.mpl_connect(
                        "button_press_event", self.mouse_clicked_vbf
                    )
                )
                self.cid.append(
                    self.fig_vbf.canvas.mpl_connect(
                        "button_release_event", self.mouse_released_vbf
                    )
                )

                self.canvas_vbf.draw()
                # get bg after initial draw
                self.background = self.fig_vbf.canvas.copy_from_bbox(
                    self.fig_vbf.axes[0].bbox
                )

                self.scene = QGraphicsScene()
                self.scene.addWidget(self.canvas_vbf)
                self.graphicsView_3.setScene(self.scene)
                self.graphicsView_3.fitInView(self.scene.sceneRect())
                # self.repaint_widget(self.graphicsView_3)
                self.update_levels_vbf()
                self.update_line(self.statusedit, "Succesfully created VBF.")
                yshap, xshap = self.vbf_data.shape
                self.update_line(self.lineEdit_10, f"Size: {xshap}x{yshap}.")
                logger.debug("Draw")
            else:
                self.vbf_im.set_array(self.vbf_data)
                self.fig_vbf.axes[0].draw_artist(self.vbf_im)

                self.crosshair_horizontal.set_alpha(0)
                self.crosshair_vertical.set_alpha(0)

                self.fig_vbf.axes[0].draw_artist(self.crosshair_horizontal)
                self.fig_vbf.axes[0].draw_artist(self.crosshair_vertical)
                self.fig_vbf.canvas.blit(self.fig_vbf.axes[0].bbox)

                self.background = self.fig_vbf.canvas.copy_from_bbox(
                    self.fig_vbf.axes[0].bbox
                )

                logger.debug("Blit")

            logger.debug(f"Callbacks: {self.fig_vbf.canvas.callbacks.callbacks}")

            # plot initial diffraction pattern (index 0)
            # self.diffraction_pattern = pg.ImageItem(image=f.get_frame(0))
            # # gl = pg.GraphicsLayout()
            # # vb = gl.addViewBox()
            # # self.graphicsView_diffraction.setCentralItem(gl)
            # # vb.addItem(self.diffraction_pattern)
            # # vb.autoRange()
            # # self.graphicsView_diffraction.addWidget(self.diffraction_pattern)
            # frame = f.get_frame(0)
            # logger.debug("frame made")
            # qi = QImage(f.shape[1], f.shape[0], QImage.Format_Grayscale16)
            # logger.debug("qimage made made")
            # qi.fromData(QByteArray(frame.tobytes()))
            # logger.debug("qbytes made")
            # scene = QGraphicsScene()
            # scene.addWidget(qi)
            # self.graphicsView_diffraction.setScene(scene)
            # self.repaint_widget(self.graphicsView_diffraction)

            if self.fig_diffraction is None:
                self.fig_diffraction = plt.figure(
                    frameon=False,
                    figsize=(
                        self.vbf_data.shape[1] / 100,
                        self.vbf_data.shape[0] / 100,
                    ),
                )

                canvas = FigureCanvas(self.fig_diffraction)
                ax = plt.Axes(self.fig_diffraction, [0.0, 0.0, 1.0, 1.0])
                ax.set_axis_off()
                self.fig_diffraction.add_axes(ax)
                self.ai_diffraction = ax.imshow(f.get_frame(0), cmap="gray")
                canvas.draw()
                self.scene_vbf = QGraphicsScene()
                self.scene_vbf.addWidget(canvas)
                self.graphicsView_diffraction.setScene(self.scene_vbf)
                self.graphicsView_diffraction.fitInView(self.scene_vbf.sceneRect())

            # add settings to hdf5 file, do this after file close
            f.close()
            rec.write_scan_parameters_hdf5(path_hdf5, **self.vbf_sets)
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")

    def mouse_enter_axes_vbf(self, event):
        self.mouse_in_axes = True

    def mouse_leave_axes_vbf(self, event):
        self.mouse_in_axes = False

    def mouse_clicked_vbf(self, event):
        if self.mouse_in_axes:
            self.mouse_pressed_vbf = True

            logger.info("clicked.")
            # restore bg
            self.fig_vbf.canvas.restore_region(self.background)

            # update artists and blit
            self.crosshair_horizontal.set_ydata((event.ydata,) * 2)
            self.crosshair_vertical.set_xdata((event.xdata,) * 2)

            self.crosshair_horizontal.set_alpha(0.5)
            self.crosshair_vertical.set_alpha(0.5)

            self.fig_vbf.axes[0].draw_artist(self.crosshair_horizontal)
            self.fig_vbf.axes[0].draw_artist(self.crosshair_vertical)
            self.fig_vbf.canvas.blit(self.fig_vbf.axes[0].bbox)
            # logger.debug(len(self.fig_vbf.axes[0].lines))

    def mouse_released_vbf(self, event):
        self.mouse_pressed_vbf = False

    def mouse_moved_update_diffraction(self, event):
        if self.mouse_in_axes and self.mouse_pressed_vbf:
            logger.info("Moved")
            # mouse in correct axes (vbf) and clicked
            # therefore draw correct diffraction frame

            # restore bg
            self.fig_vbf.canvas.restore_region(self.background)

            # update crosshairs and draw
            self.crosshair_horizontal.set_ydata((event.ydata,) * 2)
            self.crosshair_vertical.set_xdata((event.xdata,) * 2)
            self.fig_vbf.axes[0].draw_artist(self.crosshair_horizontal)
            self.fig_vbf.axes[0].draw_artist(self.crosshair_vertical)
            self.fig_vbf.canvas.blit(self.fig_vbf.axes[0].bbox)

            scan_exp_data = self.get_scan_export_data()

            # open hdf
            path_hdf5 = self.lineEdit_4.text()
            f = rec.hdf5Intermediate(path_hdf5)

            # calculate frame number
            crop = (
                int(event.xdata),
                int(event.xdata + 1),
                int(event.ydata),
                int(event.ydata + 1),
            )
            number = f.calculate_scan_export_indexes(crop=crop, **scan_exp_data)
            if len(number) != 1:
                logger.debug(f"Frame numbers: {number}")
            frame = f.get_frame(number[0])

            # possible log scaling
            if self.checkBox_diffraction_log_scale.isChecked():
                frame = np.log10(frame - frame.min() + 1)
            logger.debug(type(frame))

            self.ai_diffraction.set_array(frame)
            self.ai_diffraction.set_clim(frame.min(), frame.max())
            self.fig_diffraction.axes[0].draw_artist(self.ai_diffraction)
            self.fig_diffraction.canvas.blit(self.fig_diffraction.axes[0].bbox)
            logger.debug(
                f"No DPs in axes: {len(self.fig_diffraction.axes[0].images)}"
            )  # should be 1

            self.lineEdit_diffraction_coordinate.setText(
                f"(x, y) = ({int(event.xdata)}, {int(event.ydata)})"
            )

            self.lineEdit_diffraction_coordinate_linear.setText(f"{number[0]}")

            f.close()

    def get_scan_export_data(self):
        """The keys below should exactly match arg names for hdf5Intermediate.get_blo_export_data."""
        out = dict()
        # overwrite standard info depending on gui
        if self.checkBox_2.checkState():
            # use custom scanning
            out["sdimx"] = self.spinBox.value()
            out["sdimy"] = self.spinBox_2.value()
        else:
            out["sdimx"] = out["sdimy"] = None
        if self.checkBox_11.checkState():
            # use custom frames
            out["start_frame"] = self.spinBox_15.value()
            out["end_frame"] = self.spinBox_16.value()
        else:
            out["start_frame"] = out["end_frame"] = None
        # use hysteresis or not
        out["hyst"] = self.spinBox_9.value() if self.checkBox_6.checkState() else 0
        out[
            "hyst_dir"
        ] = self.comboBox_hyst_dir.currentText()  # irrelevant if hyst is False
        # use snake scan or not
        out["snakescan"] = self.checkBox_12.checkState()
        out[
            "snakescan_dir"
        ] = (
            self.comboBox_snakescan_dir.currentText()
        )  # irrelevant if snakescan is False

        return out

    def write_to_file(self):
        """Export the data to a specific file format in an incremental way"""
        path_hdf5 = self.lineEdit_4.text()
        path_blo = self.lineEdit_7.text()
        try:
            # check if an hdf5 file is selected
            if not path_hdf5:
                raise Exception("No valid HDF5 file selected!")
            if not path_blo:
                raise Exception("No valid export file path selected!")
            # try to read the info from the file
            f = rec.hdf5Intermediate(path_hdf5)
            # none or 0 means default

            # get all relevant scan export data
            scan_exp_data = self.get_scan_export_data()

            # we set the scale of scan and dp
            scan_scale = self.doubleSpinBox_2.value()
            dp_scale = self.doubleSpinBox_3.value()
            # calculate the image
            filetyp = self.comboBox.currentText()

            logger.debug(
                f"We try to create a {filetyp} file with data: {scan_exp_data}."
            )
            logger.debug("Calculating shape and indexes...")

            # check crop desired
            if self.checkBox_apply_crop.isChecked():
                # check crop valid
                if self.check_crop_limits() is None:
                    self.update_line(
                        self.statusedit,
                        f"File not exported, please check cropping limits.",
                    )
                    return
                # get crop
                crop = self.get_crop_limits()
            else:
                crop = None

            # export binning option setup
            binning = (
                self.spinBox_binning_blo.value()
                if self.checkBox_binning_blo.isChecked()
                else None
            )

            shape, indexes = f.get_blo_export_data(
                crop=crop, binning=binning, **scan_exp_data
            )

            logger.debug(f"Shape: {shape}")
            logger.debug(f"Starting to write {filetyp} file")
            self.update_line(self.statusedit, f"Writing {filetyp} file...")

            if filetyp == ".blo":
                self.get_thread = blf.bloFileWriter(
                    f,
                    path_blo,
                    shape,
                    indexes,
                    scan_scale,
                    dp_scale,
                    # if rescale button is checked do rescale, otherwise no rescaling selected
                    rescale=self.checkBox_rescale.isChecked(),
                    clip_max=(
                        self.spinBox_clipmax_blo.value()
                        if self.checkBox_clipmax_blo.isChecked()
                        else None
                    ),
                    binning=binning,
                    recenter=self.checkBox_center_frame.isChecked(),
                )
            elif filetyp == ".hspy":
                self.get_thread = hspf.hspyFileWriter(
                    f, path_blo, shape, indexes, scan_scale, dp_scale
                )
            else:
                raise NotImplementedError("Unrecognized file type")

            self.get_thread.increase_progress.connect(self.increase_progbar)
            self.get_thread.finish.connect(self.done_bloexport)
            self.get_thread.start()
            self.window.setEnabled(False)
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")

    def done_bloexport(self):
        self.window.setEnabled(True)
        # also update lines in the second pannel
        self.update_line(self.statusedit, "Succesfully exported to file")

    def export_tiffs(self):
        path_hdf5 = self.lineEdit_4.text()
        path_tiff = self.lineEdit_9.text()
        try:
            # check if an hdf5 file is selected
            if not path_hdf5:
                raise Exception("No valid HDF5 file selected!")
            if not path_tiff:
                raise Exception("No valid tiff file path selected!")
            pre, fin = os.path.splitext(path_tiff)

            f = rec.hdf5Intermediate(path_hdf5)
            tot_frames = f["Scan"].attrs["total_stream_frames"]
            first_frame = self.spinBox_5.value()
            last_frame = self.spinBox_4.value()
            dtype = self.comboBox_2.currentText()
            if dtype == "uint8":
                dtype = np.uint8
            elif dtype == "uint16":
                dtype = np.uint16
            else:
                raise Exception("Unexpected dtype")
            if tot_frames <= last_frame:
                raise Exception("Frames are out of range")
            frames = np.arange(first_frame, last_frame + 1)
            self.update_line(self.statusedit, "Exporting to tiff files...")
            self.get_thread = tfe.TiffFileWriter(f, frames, dtype, pre, fin)
            self.get_thread.increase_progress.connect(self.increase_progbar)
            self.get_thread.finish.connect(self.done_tiffexport)
            self.get_thread.start()
            self.window.setEnabled(False)
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")

    def done_tiffexport(self):
        self.window.setEnabled(True)
        # also update lines in the second pannel
        self.update_line(self.statusedit, "Succesfully exported Tiff files")

    def increase_progbar(self, value):
        self.progressBar.setValue(value)

    def hardRepaint(self):
        self.window.hide()
        self.window.show()

    def remove_cropped_region(self):
        # try to remove if it is there
        label = "crop_rect"
        try:
            # see if rectangle already plotted and just update
            index = [i.get_label() for i in self.fig_vbf.axes[0].patches].index(label)
            rect = self.fig_vbf.axes[0].patches[index]
            rect.remove()
            self.fig_vbf.canvas.draw()
        except ValueError:
            logger.info("No crop rectange drawn.")

    def get_crop_limits(self):
        xmin = self.spinBox_22.value()
        xmax = self.spinBox_23.value()
        ymin = self.spinBox_20.value()
        ymax = self.spinBox_21.value()

        return xmin, xmax, ymin, ymax

    def check_crop_limits(self):
        xmin, xmax, ymin, ymax = self.get_crop_limits()

        # check scan boundaries, returns None if any test fails
        if (
            xmin < 0
            or ymin < 0
            or not xmax <= self.spinBox.value()
            or not ymax <= self.spinBox_2.value()
        ):
            self.update_line(
                self.statusedit, "Crop limits should be within scan bounds."
            )
            self.remove_cropped_region()
            return

        # check cropping dimension > 0
        if not xmax - xmin > 0:
            self.update_line(
                self.statusedit,
                f"Crop dimension incorrect. Crop x: {xmax - xmin}.",
            )
            self.remove_cropped_region()
            return
        if not ymax - ymin > 0:
            self.update_line(
                self.statusedit,
                f"Crop dimension incorrect. Crop y: {ymax - ymin}.",
            )
            self.remove_cropped_region()
            return

        # return non-None if passes checks
        return True

    def show_cropped_region(self):
        if self.check_crop_limits() is None:
            return

        # check for validity of cropped region
        xmin, xmax, ymin, ymax = self.get_crop_limits()

        if self.fig_vbf is None:
            self.update_line(
                self.statusedit,
                "No VBF figure plotted yet.",
            )
            return

        label = "crop_rect"
        try:
            # see if rectangle already plotted and just update
            index = [i.get_label() for i in self.fig_vbf.axes[0].patches].index(label)
            rect = self.fig_vbf.axes[0].patches[index]
            logger.info("Updating rectangle.")
            rect.set_height(ymax - ymin)
            rect.set_width(xmax - xmin)
            rect.set_x(xmin)
            rect.set_y(ymin)
        except ValueError:
            logger.info("Plotting rectangle.")
            rect = plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                ec="k",
                ls="dashed",
                label=label,
            )
            self.fig_vbf.axes[0].add_patch(rect)
        self.fig_vbf.canvas.draw()

        self.update_line(
            self.statusedit,
            f"Crop dimensions (x, y): ({xmax - xmin}, {ymax - ymin}).",
        )


def main():
    app = QApplication([])
    window = Window()
    _ = ConnectedWidget(window)
    window.setWindowTitle("TVIPS converter")
    # window.setWindowIcon(
    #     QIcon(QPixmap(os.path.join(os.path.dirname(__file__), "Icon 128.png")))
    # )
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
