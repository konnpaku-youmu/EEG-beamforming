import sys
import random
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes1 = fig.add_subplot(231)
        self.axes2 = fig.add_subplot(232)
        self.axes3 = fig.add_subplot(233)
        self.axes4 = fig.add_subplot(234)
        self.axes5 = fig.add_subplot(235)
        self.axes6 = fig.add_subplot(236)
        super(MplCanvas, self).__init__(fig)

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # set window size
        self.setGeometry(300, 300, 1500, 900)
        self.setWindowTitle('EEG assisted acoustic zoom')

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.setCentralWidget(self.canvas)

        n_data = 50
        self.xdata = list(range(n_data))
        self.ydata = [random.randint(0, 10) for i in range(n_data)]

        self._plot_ref = None
        self.update_plot()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_plot)

        # timer for controlling the main loop
        self.tick_timer = QtCore.QTimer()
        self.tick_timer.setInterval(1000)
        self.tick_timer.timeout.connect(self.start)

        # Layouts
        # start button
        self.start_button = QtWidgets.QPushButton("Start", self)
        self.start_button.clicked.connect(self.timer.start)
        self.start_button.resize(self.start_button.sizeHint())
        self.start_button.move(1000, 40)

        # stop button
        self.stop_button = QtWidgets.QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.timer.stop)
        self.stop_button.resize(self.stop_button.sizeHint())
        self.stop_button.move(1200, 40)

        self.show()

    def start(self):
        
        pass

    def update_plot(self):
        # Drop off the first y element, append a new one.
        self.ydata = self.ydata[1:] + [random.randint(0, 10)]

        # Note: we no longer need to clear the axis.
        if self._plot_ref is None:
            # First time we have no plot reference, so do a normal plot.
            # .plot returns a list of line <reference>s, as we're
            # only getting one we can take the first element.
            plot_refs = self.canvas.axes1.plot(self.xdata, self.ydata, 'r')
            self._plot_ref = plot_refs[0]
        else:
            # We have a reference, we can use it to update the data for that line.
            self._plot_ref.set_ydata(self.ydata)

        # Trigger the canvas to update and redraw.
        self.canvas.draw()


app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
app.exec_()