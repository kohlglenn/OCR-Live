import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from src.SavedModel import get_model

matplotlib.use('Qt5Agg')
REFRESH_RATE = 100


def pixels_to_inches(pixels, dpi):
    return pixels/dpi


def inches_to_pixels(inches, dpi):
    return inches * dpi


class MplCanvas(FigureCanvasQTAgg):
    # Fig size is in inches, constructor is in pixels
    def __init__(self, parent=None, width=280, height=280, dpi=100):
        w, h = pixels_to_inches(width, dpi), pixels_to_inches(height, dpi)
        self.fig = Figure(figsize=(w, h), dpi=dpi)
        self.axes = self.fig.add_subplot(1, 1, 1)
        super(MplCanvas, self).__init__(self.fig)

    def plot(self, probabilities):
        self.axes.clear()
        self.axes.barh([x for x in range(0, 10)], probabilities)
        self.axes.set_yticks(np.arange(10))
        self.fig.canvas.draw()


class Canvas(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        pixmap = QtGui.QPixmap(280, 280)
        pixmap.fill(QtGui.QColor("#FFFFFF"))
        self.setPixmap(pixmap)

        self.last_x, self.last_y = None, None
        self.pen_color = QtGui.QColor('#000000')
        self.changed = False

    def reset(self):
        pixmap = QtGui.QPixmap(280, 280)
        pixmap.fill(QtGui.QColor("#FFFFFF"))
        self.setPixmap(pixmap)

    def set_pen_color(self, c):
        self.pen_color = QtGui.QColor(c)

    def mouseMoveEvent(self, e):
        if self.last_x is None:  # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return  # Ignore the first time.

        painter = QtGui.QPainter(self.pixmap())
        p = painter.pen()
        p.setWidth(20)
        p.setColor(self.pen_color)
        painter.setPen(p)
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

        self.changed = True

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR Live Demo")
        self.setWindowIcon(QIcon("../images/py.png"))

        cl = QtWidgets.QLabel()
        cl.setText("?")
        cl.setFont(QFont("Arial", 70, QtGui.QFont.Bold))
        cl.setAlignment(Qt.AlignCenter)
        self.class_label = cl
        c = QtWidgets.QLabel()
        c.setText("Classification:")
        c.setFont(QFont("Arial", 22, QtGui.QFont.Bold))
        c.setAlignment(Qt.AlignCenter)
        reset = QtWidgets.QPushButton()
        reset.setText("Click to reset drawing")
        reset.setFont(QFont("Arial", 22))

        div1 = QtWidgets.QVBoxLayout()
        div1.addWidget(c)
        div1.addWidget(self.class_label)
        div1.addWidget(reset)

        self.canvas = Canvas()
        # reset.pressed.connect(self.get_predictions)
        reset.pressed.connect(self.reset)
        sc = MplCanvas(self, width=280, height=280, dpi=100)
        sc.plot([0 for x in range(0, 10)])
        self.chart = sc

        div2 = QtWidgets.QHBoxLayout()
        div2.addWidget(self.canvas)
        div2.addWidget(self.chart)
        div2.addLayout(div1)

        palette = QtWidgets.QHBoxLayout()
        self.add_palette_buttons(palette)

        div3 = QtWidgets.QVBoxLayout()
        div3.addLayout(div2)
        div3.addLayout(palette)
        w = QtWidgets.QWidget()
        w.setLayout(div3)

        self.setCentralWidget(w)
        self.setGeometry(50, 50, 900, 320)
        self.show()
        self.model = get_model()
        self._timer_painter = QTimer(self)
        self._timer_painter.start(REFRESH_RATE)
        self._timer_painter.timeout.connect(self.get_predictions)

    def add_palette_buttons(self, layout):
        for c in COLORS:
            b = QPaletteButton(c)
            b.pressed.connect(lambda c=c: self.canvas.set_pen_color(c))
            layout.addWidget(b)

    def get_predictions(self):
        if not self.canvas.changed:
            return
        p = self.canvas.pixmap()
        scaled = p.scaled(28, 28, Qt.KeepAspectRatio)
        channels_count = 4
        image = scaled.toImage()
        b = image.bits()
        # sip.voidptr must know size to support python buffer interface
        b.setsize(28 * 28 * channels_count)
        arr = np.frombuffer(b, np.uint8).reshape((28, 28, channels_count))
        gs_arr = rgba_to_inv_grayscale(arr)
        gs_arr = np.array(gs_arr).reshape(1, 28, 28, 1)
        predictions = self.model.predict(gs_arr)
        self.chart.plot(predictions[0])
        idx, tmp = -1, -1
        for i in range(0, 10):
            if predictions[0][i] > tmp:
                idx, tmp = i, predictions[0][i]
        self.class_label.setNum(idx)
        self.class_label.update()
        self.canvas.changed = False

    def reset(self):
        self.canvas.reset()
        self.chart.plot([0 for x in range(0, 10)])
        self.class_label.setText("?")
        self.class_label.update()


COLORS = [
    # 17 undertones https://lospec.com/palette-list/17undertones
    '#000000', '#141923', '#414168', '#3a7fa7', '#35e3e3', '#8fd970', '#5ebb49',
    '#458352', '#dcd37b', '#fffee5', '#ffd035', '#cc9245', '#a15c3e', '#a42f3b',
    '#f45b7a', '#c24998', '#81588d', '#bcb0c2', '#ffffff',
]


class QPaletteButton(QtWidgets.QPushButton):

    def __init__(self, color):
        super().__init__()
        self.setFixedSize(QtCore.QSize(24, 24))
        self.color = color
        self.setStyleSheet("background-color: %s;" % color)


# Returns array with dimensions [w][h][1] = 255 - avg(RGB channels) to be in same
# format as the data set
def rgba_to_inv_grayscale(arr):
    ret = [[[my_fun(arr[idx1][idx2])] for idx2 in range(0, len(arr[0]))] for idx1 in range(0, len(arr))]
    return ret


def my_fun(arr):
    return 255 - (arr[0]/3 + arr[1]/3 + arr[2]/3)


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
app.exec_()
