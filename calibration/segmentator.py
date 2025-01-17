import os
import sys
import glob
import time
import joblib
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


def load_image(filename):
    img = pyplot.imread(filename)[:, :, :3]
    return (255 * img).astype(np.uint8) if img.dtype == np.dtype(np.float32) else np.copy(img)


def apply_mask(img, mask):
    if mask is not None:
        r, c = np.nonzero(mask)
        img[r, c, :] = [0, 255, 255]
    return img


def ensure_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


class QFigure(QtWidgets.QWidget):
    def __init__(self, title='', toolbar=True, show=False):
        super().__init__()
        self.setMinimumWidth(320)
        self.setMinimumHeight(160)
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
            self.setAttribute(QtCore.Qt.WA_QuitOn180Close, False)
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
        self.img_view, self.colors_view = None, None
        self.sz, self.mode, self.button = 100, "", None

        self.fimg = QFigure('img')
        self.fcolors = QFigure('colors', toolbar=False)
        self.flayout = QtWidgets.QVBoxLayout()
        self.flayout.addWidget(self.fimg, 19)
        self.flayout.addWidget(self.fcolors, 1)
        self.ui.layout.addLayout(self.flayout, 9)
        self.fimg.canvas.mpl_connect('button_press_event', self.on_image_press)
        self.fimg.canvas.mpl_connect('button_release_event', self.on_image_release)
        self.fimg.canvas.mpl_connect('motion_notify_event', self.on_image_move)
        self.fcolors.canvas.mpl_connect('button_press_event', self.on_color_click)
        self.ui.filenameCB.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        self.choose = functools.partial(self.wrap, self.on_choose_folder, "Choose folder")
        self.locate = functools.partial(self.wrap, self.on_locate_masks, "Locate masks")
        self.search = functools.partial(self.wrap, self.on_search_files, "Search files")
        self.select = functools.partial(self.wrap, self.on_select_file, "Select file")
        self.quantize = functools.partial(self.wrap, self.on_quantize, "Quantize")
        self.assign = functools.partial(self.wrap, self.on_assign, "Assign")
        self.save = functools.partial(self.wrap, self.on_save, "Save")

        self.ui.actionChooseFolder.triggered.connect(self.choose)
        self.ui.actionLocateMasks.triggered.connect(self.locate)
        self.ui.actionSearchFiles.triggered.connect(self.search)
        self.ui.actionQuantize.triggered.connect(self.quantize)
        self.ui.actionAssign.triggered.connect(self.assign)
        self.ui.actionSave.triggered.connect(self.save)

        self.ui.choosePB.clicked.connect(self.choose)
        self.ui.locatePB.clicked.connect(self.locate)
        self.ui.searchPB.clicked.connect(self.search)
        self.ui.quantizePB.clicked.connect(self.quantize)
        self.ui.assignPB.clicked.connect(self.assign)
        self.ui.savePB.clicked.connect(self.save)

        self.ui.folderLE.returnPressed.connect(self.search)
        self.ui.masksLE.returnPressed.connect(self.search)
        self.ui.templateLE.returnPressed.connect(self.search)
        self.ui.filenameCB.currentTextChanged.connect(self.select)
        self.ui.applyMaskCB.stateChanged.connect(self.select)
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

    def on_locate_masks(self):
        self.mode = ""
        new_dir = QtWidgets.QFileDialog.getExistingDirectory(self.ui, 'Locate masks', self.ui.masksLE.text())

        if os.path.isdir(new_dir):
            self.ui.masksLE.setText(new_dir)

            return "New location \'%s\'" % new_dir
        else:
            raise ValueError("Invalid location: \'%s\'" % new_dir)

    def on_search_files(self):
        self.mode = ""
        self.folder = self.ui.folderLE.text()
        self.template = self.ui.templateLE.text()

        self.ui.filenameCB.clear()
        filenames = sorted(glob.glob(self.folder + "/" + self.template))

        print(len(filenames), "files:", filenames)

        if len(filenames) == 0:
            raise RuntimeError("No files found")

        for filename in filenames:
            self.ui.filenameCB.addItem(os.path.basename(filename))

        return "%d files found" % len(filenames)

    def on_select_file(self):
        self.mode = ""
        self.filename = self.ui.filenameCB.currentText()
        print("Image:", self.filename)

        path = self.folder + "/" + self.filename
        if self.filename == "" or not os.path.exists(path):
            return "No file selected"

        candidates = [self.folder + "/" + self.filename[:-4] + "_mask.png",
                      self.folder + "/" + self.filename[:-4] + "_mask.jpg",
                      self.ui.masksLE.text() + "/" + self.filename[:-4] + ".png",
                      self.ui.masksLE.text() + "/" + self.filename[:-4] + ".jpg"]

        candidates.extend([c[:-4] + "_" + c[-4:] for c in candidates])

        mask_name, self.mask = "", None
        for candidate in candidates:
            if os.path.exists(candidate):
                mask_name = candidate
                break

        if os.path.exists(mask_name):
            self.mask = np.sum(load_image(mask_name), axis=2) > 0
            print("Mask:", mask_name)

        self.img = load_image(path)

        if self.ui.applyMaskCB.isChecked():
            self.img = apply_mask(self.img, self.mask)

        if self.ui.autoQuantizeCB.isChecked():
            self.quantize()
        else:
            self.fcolors.clear(cd=False)
            self.fcolors.redraw()

            self.fimg.clear()
            self.fimg.ax.imshow(self.img)
            self.fimg.fig.tight_layout()
            self.fimg.redraw()

        return self.filename + " selected" + (" (with mask)" if self.mask is not None else "")

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

    def plot_img(self, fig, img, view=None):
        if view is not None:
            view.set_data(img)
            fig.redraw()
        else:
            fig.clear()
            if img is not None:
                view = fig.ax.imshow(img)
                fig.fig.tight_layout()
                fig.redraw()
        return view

    def plot_mapped(self, from_scratch=True):
        self.mapped = self.map(self.labels, self.mapped_colors)
        self.palette = self.gen_palette(self.mapped_colors)

        self.img_view = self.plot_img(self.fimg, self.mapped, None if from_scratch else self.img_view)
        self.colors_view = self.plot_img(self.fcolors, self.palette, None if from_scratch else self.colors_view)

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
            elif self.shift:
                self.mapped_colors[i, :] = [255, 255, 0]
                self.mapping[i] = 6

        if button == MOUSE_MIDDLE:
            if self.ctrl:
                self.mapped_colors[i, :] = [0, 0, 255]
                self.mapping[i] = 3
            elif self.alt:
                self.mapped_colors[i, :] = [255, 0, 255]
                self.mapping[i] = 5
            elif self.shift:
                self.mapped_colors[i, :] = [0, 255, 255]
                self.mapping[i] = 7

        if button == MOUSE_RIGHT:
            if self.alt:
                self.mapped_colors[i, :] = [0, 0, 0]
                self.mapping[i] = 0
            elif self.ctrl:
                self.mapped_colors[i, :] = [255, 0, 0]
                self.mapping[i] = 1
            elif self.shift:
                self.mapped_colors[i, :] = [0, 0, 0]
                self.mapping[i] = 0

    def on_image_press(self, e):
        self.button = e.button
        self.on_image_click(e)

    def on_image_release(self, e):
        self.button = None
        self.on_image_click(e)

    def on_image_move(self, e):
        e.button = self.button
        self.on_image_click(e)

    def on_image_click(self, e):
        self.update_modifiers()
        # print("Image:", e.xdata, e.ydata, e.button, self.ctrl, self.shift, self.alt)

        if e.xdata and e.ydata and (self.ctrl or self.shift or self.alt) and self.mode != "":
            r, c = int(e.ydata + 0.5), int(e.xdata + 0.5)

            if self.mode == "mapping":
                color = self.quantized[r, c]
                i = np.argmin(np.sum(np.abs(self.colors - color), axis=1))
            else:
                i = self.labels.reshape(self.img.shape[:2])[r, c]
                color = self.colors[i, :]

            # print(i, color, self.colors[i, :])

            self.map_color(i, e.button)

            if self.button is None:
                self.plot_mapped(from_scratch=False)

    def on_color_click(self, e):
        self.update_modifiers()
        # print("Color:", e.xdata, e.ydata, e.button, self.ctrl, self.shift, self.alt)

        if e.xdata and e.ydata and (self.ctrl or self.shift or self.alt) and self.mode == "mapping":
            i = int(e.xdata + 0.5) // self.sz
            color = self.colors[i, :]
            # print(i, color)

            self.map_color(i, e.button)
            self.plot_mapped(from_scratch=False)

    def to_gray(self, img):
        img[np.nonzero(img)] = 1
        img *= np.array([1, 2, 4], dtype=np.uint8)[None, None, :]

        gray = np.zeros(img.shape[:2], dtype=np.uint8)
        gray[...] = np.sum(img, axis=2, dtype=np.uint8)
        correspondence = np.array([0, 1, 2, 6, 3, 5, 7, 4], dtype=np.uint8)
        gray[...] = correspondence[gray.ravel()].reshape(img.shape[:2])

        return gray

    def from_gray(self, gray):
        img = np.zeros((*gray.shape, 3), dtype=np.uint8)
        colors = np.array([[0, 0, 0],
                           [255, 0, 0],
                           [0, 255, 0],
                           [0, 0, 255],
                           [255, 255, 255],
                           [255, 0, 255],
                           [255, 255, 0],
                           [0, 255, 255]], dtype=np.uint8)

        img.reshape((-1, 3))[...] = colors[gray.ravel(), :]

        return img

    def on_assign(self):
        self.mode = ""

        # Map remaining colors
        for i in range(self.n):
            if self.mapping[i] == 0:
                self.mapped_colors[i, :] = [0, 0, 0]
            # elif self.mapping[i] == 3:
            #     self.mapped_colors[i, :] = [255, 255, 255]
            # elif self.mapping[i] == 7:
            #     self.mapped_colors[i, :] = [255, 255, 255]
            # elif self.mapping[i] == 4:
            #     self.mapped_colors[i, :] = [0, 0, 0]

        self.gray = self.to_gray(self.map(self.labels, self.mapped_colors))

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

        self.assigned = self.from_gray(self.gray)

        # Disable background feature with non-achievable gray value 8
        new_labels, m = measure.label(self.gray, background=8, return_num=True)
        self.labels, self.n, self.sz = new_labels.ravel(), m + 1, 1
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

        # self.mapped = apply_mask(self.mapped, self.mask)
        self.plot_img(self.fimg, self.mapped)

        pyplot.imsave(new_filename, self.mapped)
        return "Saved " + new_filename


def select_single(img_in, img_out, mask_in, mask_out, has_mask=False):
    pyplot.imsave(img_out, load_image(img_in))

    if has_mask:
        pyplot.imsave(mask_out, load_image(mask_in))


def select_all(mask_mandatory=True):
    all_images_path = "../all_images/"
    all_masks_path = "../all_masks/"
    selected_images_path = "../selected_images/"
    selected_masks_path = "../selected_masks/"

    img_names = glob.glob(all_images_path + "*.jpg")
    mask_names = glob.glob(all_masks_path + "*.jpg")
    img_names.extend(glob.glob(all_images_path + "*.png"))
    mask_names.extend(glob.glob(all_masks_path + "*.png"))
    img_names, mask_names = sorted(img_names), sorted(mask_names)

    jobs, j, j_tot = [], 0, 0
    for i, img_name in enumerate(img_names):
        # if j_tot > 150:
        #     break

        j = j_tot // 100
        ensure_exists(selected_images_path + str(j))
        ensure_exists(selected_masks_path + str(j))

        base_name = os.path.basename(img_name)
        candidates = [all_masks_path + base_name[:-3] + ext for ext in ["png", "jpg"]]
        candidates.extend([c[:-4] + "_" + c[-4:] for c in candidates])

        has_mask, mask_name = False, ""
        for candidate in candidates:
            if os.path.exists(candidate):
                has_mask, mask_name = True, candidate
                break

        if mask_mandatory and not has_mask:
            print(i, base_name, "skipped in", j)
            continue
        else:
            print(i, base_name, "copying in", j)

        jobs.append(joblib.delayed(select_single, check_pickle=False)
                    (img_name, selected_images_path + str(j) + "/" + base_name[:-4] + ".png",
                     mask_name, selected_masks_path + str(j) + "/" + base_name[:-4] + ".png",
                     has_mask=has_mask))
        j_tot += 1

    print(len(jobs), "jobs")
    joblib.Parallel(verbose=15, n_jobs=-1, batch_size=1, pre_dispatch="all")(jobs)


if __name__ == '__main__':
    # select_all(mask_mandatory=True)
    # exit()

    app = QtWidgets.QApplication(sys.argv)

    gui = GUI()
    gui.ui.show()

    sys.exit(app.exec_())
