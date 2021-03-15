import os
import sys
import glob
import time
import functools
import numpy as np
from matplotlib import pyplot
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from skimage import measure
from skimage import filters

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5 import uic, QtCore, QtWidgets

MOUSE_LEFT = 1
MOUSE_MIDDLE = 2
MOUSE_RIGHT = 3


class QFigure(QtWidgets.QWidget):
    def __init__(self, title='', toolbar=True, show=False):
        super().__init__()
        self.setMinimumWidth(480)
        self.setMinimumHeight(240)
        self.fig = Figure()
        self.ax = None
        self.cd()

        self.canvas = FigureCanvas(self.fig)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.canvas)

        if toolbar:
            self.toolbar = NavigationToolbar(self.canvas, self.canvas)
            self.layout.addWidget(self.toolbar)

        self.setLayout(self.layout)
        self.setWindowTitle(title)

        if show:
            # Set so that Figure windows doesn't prevent the app termination when Main window is closed
            self.setAttribute(QtCore.Qt.WA_QuitOnClose, False)
            self.show()

    def clear(self, cd=True):
        self.fig.clf()

        if cd:
            self.cd()

    def cd(self, *args, **kwargs):
        if len(args):
            self.ax = self.fig.add_subplot(*args, **kwargs)
        else:
            self.ax = self.fig.add_subplot(111)

    def redraw(self):
        self.canvas.draw()


class GUI():
    def __init__(self):
        self.ui = uic.loadUi('segmentator.ui')
        self.ui.resize(1600, 900)
        self.ctrl, self.shift, self.alt = False, False, False
        self.sz, self.mode = 100, ""

        self.fimg = QFigure('img')
        self.fcolors = QFigure('colors', toolbar=False)
        self.flayout = QtWidgets.QVBoxLayout()
        self.flayout.addWidget(self.fimg, 6)
        self.flayout.addWidget(self.fcolors, 1)
        self.ui.layout.addLayout(self.flayout, 6)
        self.fimg.canvas.mpl_connect('button_press_event', self.on_image_click)
        self.fcolors.canvas.mpl_connect('button_press_event', self.on_color_click)

        self.choose = functools.partial(self.wrap, self.on_choose_folder, "Choose folder")
        self.search = functools.partial(self.wrap, self.on_search_files, "Search files")
        self.select = functools.partial(self.wrap, self.on_select_file, "Select file")
        self.quantize = functools.partial(self.wrap, self.on_quantize, "Quantize")
        self.assign = functools.partial(self.wrap, self.on_assign, "Assign")
        self.save = functools.partial(self.wrap, self.on_save, "Save")

        self.ui.actionChooseFolder.triggered.connect(self.choose)
        self.ui.actionSearchFiles.triggered.connect(self.search)
        self.ui.actionQuantize.triggered.connect(self.quantize)
        self.ui.actionAssign.triggered.connect(self.assign)
        self.ui.actionSave.triggered.connect(self.save)

        self.ui.choosePB.clicked.connect(self.choose)
        self.ui.searchPB.clicked.connect(self.search)
        self.ui.quantizePB.clicked.connect(self.quantize)
        self.ui.assignPB.clicked.connect(self.assign)
        self.ui.savePB.clicked.connect(self.save)

        self.ui.folderLE.returnPressed.connect(self.search)
        self.ui.templateLE.returnPressed.connect(self.search)
        self.ui.filenameCB.currentTextChanged.connect(self.select)
        self.ui.numColorsSP.valueChanged.connect(self.quantize)
        self.ui.suffixLE.returnPressed.connect(self.save)

        self.search()

    def wrap(self, job, title):
        t, err = time.time(), None

        try:
            info = job()
        except Exception as e:
            err = e

        if err:
            msg = "%s failed in %.3f seconds. Reason: %s" % (title, time.time() - t, str(err))
        else:
            msg = "%s done in %.3f seconds. %s" % (title, time.time() - t, info or "")

        self.ui.statusBar().setStyleSheet("color: " + ("red" if err else "green"))
        self.ui.statusBar().showMessage(msg)

        if err:
            self.notify(QtWidgets.QMessageBox.Warning, title + " failed!", str(err))

    def status(self, msg):
        self.ui.statusBar().setStyleSheet("color: black")
        self.ui.statusBar().showMessage(msg)
        print(msg)

    def notify(self, icon, title, description):
        messageBox = QtWidgets.QMessageBox(self.ui)
        messageBox.setIcon(icon)
        messageBox.setWindowTitle(title)
        messageBox.setText(description)
        messageBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        messageBox.open()

    def update_modifiers(self):
        self.ctrl = QtWidgets.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier
        self.shift = QtWidgets.QApplication.keyboardModifiers() == QtCore.Qt.ShiftModifier
        self.alt = QtWidgets.QApplication.keyboardModifiers() == QtCore.Qt.AltModifier

    def on_choose_folder(self):
        self.mode = ""
        new_dir = QtWidgets.QFileDialog.getExistingDirectory(self.ui, 'Select working directory', self.ui.folderLE.text())

        if os.path.isdir(new_dir):
            self.ui.folderLE.setText(new_dir)

            if self.ui.autoSearchCB.isChecked():
                self.search()

            return "New folder \'%s\'" % new_dir
        else:
            raise ValueError("Invalid directory: \'%s\'" % new_dir)

    def on_search_files(self):
        self.mode = ""
        self.folder = self.ui.folderLE.text()
        self.template = self.ui.templateLE.text()

        self.ui.filenameCB.clear()
        filenames = glob.glob(self.folder + "/" + self.template)

        print(len(filenames), "files:", filenames)

        if len(filenames) == 0:
            raise RuntimeError("No files found")

        for filename in filenames:
            self.ui.filenameCB.addItem(os.path.basename(filename))

        return "%d files found" % len(filenames)

    def on_select_file(self):
        self.mode = ""
        self.filename = self.ui.filenameCB.currentText()
        print(self.filename)

        path = self.folder + "/" + self.filename
        if self.filename == "" or not os.path.exists(path):
            return "No file selected"

        self.img = pyplot.imread(path)

        if self.ui.autoQuantizeCB.isChecked():
            self.quantize()
        else:
            self.fcolors.clear(cd=False)
            self.fcolors.redraw()

            self.fimg.clear()
            self.fimg.ax.imshow(self.img)
            self.fimg.fig.tight_layout()
            self.fimg.redraw()

        return self.filename + " selected"

    def map(self, labels, colors):
        img = np.zeros_like(self.img)
        img[:] = colors[labels, :].reshape(self.img.shape)
        return img

    def gen_palette(self, colors, sz=None):
        n, sz = colors.shape[0], sz or self.sz
        img = np.zeros((sz, n * sz, 3), dtype=np.uint8)

        for i in range(n):
            img[:, i*sz:(i+1)*sz] = colors[i, :]

        return img

    def plot_img(self, fig, img):
        fig.clear()
        if img is not None:
            fig.ax.imshow(img)
            fig.fig.tight_layout()
            fig.redraw()

    def plot_mapped(self):
        self.mapped = self.map(self.labels, self.mapped_colors)
        self.palette = self.gen_palette(self.mapped_colors)

        self.plot_img(self.fimg, self.mapped)
        self.plot_img(self.fcolors, self.palette)

    def on_quantize(self):
        self.mode = ""
        self.n = self.ui.numColorsSP.value()

        flat = self.img.reshape(-1, 3)
        samples = shuffle(flat, n_samples=1000, random_state=0)
        kmeans = KMeans(n_clusters=self.n, random_state=0).fit(samples)
        self.colors = kmeans.cluster_centers_.astype(np.uint8)
        self.labels = kmeans.predict(flat)
        self.quantized = self.map(self.labels, self.colors)
        self.sz = 100

        self.mapped_colors = np.copy(self.colors)
        self.mapping = np.zeros(self.n)
        self.plot_mapped()

        self.mode = "mapping"
        return "Quantized %s into %d colors" % (self.filename, self.n)

    def map_color(self, i, button):
        if button == MOUSE_LEFT:
            if self.alt:
                self.mapped_colors[i, :] = [255, 255, 255]
                self.mapping[i] = 4
            elif self.ctrl:
                self.mapped_colors[i, :] = [0, 255, 0]
                self.mapping[i] = 2

        if button == MOUSE_MIDDLE:
            if self.ctrl:
                self.mapped_colors[i, :] = [0, 0, 255]
                self.mapping[i] = 3
            elif self.alt:
                self.mapped_colors[i, :] = [255, 0, 255]
                self.mapping[i] = 5

        if button == MOUSE_RIGHT:
            if self.alt:
                self.mapped_colors[i, :] = [0, 0, 0]
                self.mapping[i] = 0
            elif self.ctrl:
                self.mapped_colors[i, :] = [255, 0, 0]
                self.mapping[i] = 1

    def on_image_click(self, e):
        self.update_modifiers()
        print("Image:", e.xdata, e.ydata, e.button, self.ctrl, self.shift, self.alt)

        if e.xdata and e.ydata and (self.ctrl or self.alt) and self.mode != "":
            r, c = int(e.ydata), int(e.xdata)

            if self.mode == "mapping":
                color = self.quantized[r, c]
                i = np.argmin(np.sum(np.abs(self.colors - color), axis=1))
            else:
                i = self.labels.reshape(self.img.shape[:2])[r, c]
                color = self.colors[i, :]

            print(i, color, self.colors[i, :])

            self.map_color(i, e.button)
            self.plot_mapped()

    def on_color_click(self, e):
        self.update_modifiers()
        print("Color:", e.xdata, e.ydata, e.button, self.ctrl, self.shift, self.alt)

        if e.xdata and e.ydata and (self.ctrl or self.alt) and self.mode == "mapping":
            i = int(e.xdata) // self.sz
            color = self.colors[i, :]
            print(i, color)

            self.map_color(i, e.button)
            self.plot_mapped()

    def to_gray(self, img):
        img[np.nonzero(img)] = 1
        img *= np.array([1, 2, 3], dtype=np.uint8)[None, None, :]

        gray = np.zeros(img.shape[:2], dtype=np.uint8)
        gray[...] = np.sum(img, axis=2, dtype=np.uint8)
        gray[np.equal(gray, 4)] = 5
        gray[np.equal(gray, 6)] = 4

        return gray

    def from_gray(self, gray):
        img = np.zeros((*gray.shape, 3), dtype=np.uint8)
        colors = np.array([[0, 0, 0],
                           [255, 0, 0],
                           [0, 255, 0],
                           [0, 0, 255],
                           [255, 255, 255],
                           [255, 0, 255]], dtype=np.uint8)

        img.reshape((-1, 3))[...] = colors[gray.ravel(), :]

        return img

    def on_assign(self):
        self.mode = ""

        # Map remaining colors
        for i in range(self.n):
            if self.mapping[i] == 0:
                self.mapped_colors[i, :] = [0, 0, 0]
            elif self.mapping[i] == 3:
                self.mapped_colors[i, :] = [255, 255, 255]
            elif self.mapping[i] == 4:
                self.mapped_colors[i, :] = [0, 0, 0]

        self.assigned = self.map(self.labels, self.mapped_colors)
        self.gray = self.to_gray(np.copy(self.assigned))

        if self.ui.majorityVoteCB.isChecked():
            # for i in range(2 if self.ui.twiceCB.isChecked() else 1):
            counts = filters.rank.windowed_histogram(self.gray, np.ones((3, 3), dtype=bool))
            major = counts.argmax(axis=-1)

            res, n = counts.shape[0] * counts.shape[1], counts.shape[2]
            counts_v = counts.reshape((-1, n))
            print(counts.shape, res, n)

            idx = counts_v[np.arange(res), self.gray.ravel()] < counts_v[np.arange(res), major.ravel()]
            self.gray.reshape((-1))[idx] = major.ravel()[idx]

            # self.plot_img(self.fimg, self.gray)

        new_labels, m = measure.label(self.gray, return_num=True)
        self.labels, self.n, self.sz = new_labels.ravel(), m + 1, 2
        self.colors = np.zeros((self.n, 3), dtype=np.uint8)

        props = measure.regionprops(new_labels)
        for i in range(len(props)):
            l = props[i]['label']
            r, c = props[i]['coords'][0, :]
            self.colors[l, :] = self.assigned[r, c, :] / 2

        self.mapped_colors = np.copy(self.colors)
        self.mapping = np.zeros(self.n)
        self.plot_mapped()

        self.mode = "masking"
        return "%d regions after assignment" % len(props)

    def on_save(self):
        self.mode = ""
        self.suffix = self.ui.suffixLE.text()
        new_filename = self.folder + "/" + self.filename[:-4] + self.suffix + ".png"

        self.mapped[self.mapped == 127] = 0

        if self.ui.majorityVoteCB.isChecked():
            gray = None
            for i in range(2 if self.ui.twiceCB.isChecked() else 1):
                if gray is None:
                    gray = self.to_gray(np.copy(self.mapped))

                mask = np.ones((3, 3), dtype=bool)
                mask[0, 0] = mask[0, 2] = mask[2, 0] = mask[2, 2] = False
                counts = filters.rank.windowed_histogram(gray, mask)
                major = counts.argmax(axis=-1)

                res, n = counts.shape[0] * counts.shape[1], counts.shape[2]
                counts_v = counts.reshape((-1, n))
                print(counts.shape, res, n)

                idx = counts_v[np.arange(res), gray.ravel()] < counts_v[np.arange(res), major.ravel()]
                gray.reshape((-1))[idx] = major.ravel()[idx]

            # self.plot_img(self.fimg, self.gray)
            self.mapped = self.from_gray(gray)

        self.plot_img(self.fimg, self.mapped)

        pyplot.imsave(new_filename, self.mapped)
        return "Saved " + new_filename


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    gui = GUI()
    gui.ui.show()

    sys.exit(app.exec_())
