from cProfile import label
from util_tools import *
from FD_GSC import *

import sys
import random
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtWidgets, QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import keras

def read_eegdata(sub_num, trial_num):
    eeg_data = sio.loadmat(
        "eeg_data/eeg_data/{0}/{1}/eeg.mat".format(sub_num, trial_num))
    eeg_data = eeg_data['eeg']

    env1_data = sio.loadmat(
        "eeg_data/eeg_data/{0}/{1}/env1.mat".format(sub_num, trial_num))
    env1_data = env1_data['env1']

    env2_data = sio.loadmat(
        "eeg_data/eeg_data/{0}/{1}/env2.mat".format(sub_num, trial_num))
    env2_data = env2_data['env2']

    mapping = sio.loadmat(
        "eeg_data/eeg_data/{0}/{1}/mapping.mat".format(sub_num, trial_num))
    mapping = mapping['mapping'][0]

    locus_data = sio.loadmat(
        "eeg_data/eeg_data/{0}/{1}/Locus.mat".format(sub_num, trial_num))
    locus_data = list(map(lambda x: mapping.index(x), locus_data['locus']))

    return eeg_data, env1_data, env2_data, locus_data, mapping

class FD_GSC_EEG:
    def __init__(self, sub_num, trial_num):
        self.RIRs = RIR_Info("Data/RIRs_16kHz/LMA_rirs.mat")
        self.sub_num = sub_num
        self.trial_num = trial_num
        self.model = keras.models.load_model("./models/fine-tuned models_LSTM_5sec_64hz/LSTM_sub{}.h5".format(sub_num))
        self.speech = []
        self.step_idx = 0
        self.DOA_attention = 0
        self.hit = False
        self.attention = None
    
    def load_data(self):
        self.eeg_data, self.env1_data, self.env2_data, self.locus_data, self.mapping = read_eegdata(self.sub_num, self.trial_num)
        self.fs, self.audio = load_audios(self.sub_num, self.trial_num)

        self.eeg_wsize = self.eeg_data.shape[0]
        self.win_cnt = self.eeg_data.shape[-1]
        self.audio_wsize = self.fs * 5
    
    def step(self):
        eeg = self.eeg_data[:, :, self.step_idx]
        env1 = self.env1_data[:, :, self.step_idx]
        env2 = self.env2_data[:, :, self.step_idx]
        locus_seg = self.locus_data[self.step_idx*self.eeg_wsize:(self.step_idx+1)*self.eeg_wsize]
        locus_label = np.round(np.mean(locus_seg))
        eeg = np.expand_dims(eeg, 0)
        env1 = np.expand_dims(env1, 0)
        env2 = np.expand_dims(env2, 0)

        ynew = int(self.model.predict([eeg, env1, env2])[0, 0] > 0.5)

        self.attention = self.mapping[ynew]

        self.hit = (ynew == locus_label)

        print("Attention: {0}\t Hit={1}".format(
            self.attention, ynew == locus_label))

        # DOA
        audio_seg = self.audio[self.step_idx*self.audio_wsize:(self.step_idx+1)*self.audio_wsize, :]
        theta, pseudo_spectrum, peaks = doa_esti_MUSIC(audio_seg, self.RIRs)

        # plt.plot(pseudo_spectrum)
        # plt.show()

        DOA_esti = theta[peaks] - 90
        
        if 'F' not in self.mapping:
            if self.attention == 'L':
                self.DOA_attention = min(DOA_esti)
            elif self.attention == 'R':
                self.DOA_attention = max(DOA_esti)
        elif 'L' not in self.mapping:
            if self.attention == 'F':
                self.DOA_attention = min(abs(DOA_esti))
            elif self.attention == 'R':
                self.DOA_attention = max(DOA_esti)
        elif 'R' not in self.mapping:
            if self.attention == 'L':
                self.DOA_attention = min(DOA_esti)
            elif self.attention == 'F':
                self.DOA_attention = min(abs(DOA_esti))

        print("DOA: {0}".format(DOA_esti))
        print("DOA_attention: {0}".format(self.DOA_attention))
        rir_idx = np.argmin(abs(self.RIRs.RIR_angles - self.DOA_attention))

        speeches = DAS(audio_seg, self.DOA_attention, self.RIRs)

        # _, speeches = fd_gsc(audio_seg, self.RIRs, rir_idx=rir_idx)
        # speeches = speeches[0]

        speeches = speeches / np.max(np.abs(speeches))

        self.speech.append(speeches)
        self.step_idx += 1

        if self.step_idx == self.win_cnt:
            self.step_idx = 0
            self.speech = np.concatenate(self.speech, axis=0)
            wavfile.write("./audio_outputs/speech_est.wav", self.RIRs.sample_rate, self.speech)
            self.speech = []
            return 1
        
        return 0
    
    def get_speech(self):
        if self.speech == []:
            return np.zeros((self.audio_wsize, 1))
        else:
            return np.concatenate(self.speech, axis=0)

    def run_all(self):
        for seg in range(0, 6):
            eeg = self.eeg_data[:, :, seg]
            env1 = self.env1_data[:, :, seg]
            env2 = self.env2_data[:, :, seg]
            locus_seg = self.locus_data[seg*self.eeg_wsize:(seg+1)*self.eeg_wsize]
            locus_label = np.round(np.mean(locus_seg))
            eeg = np.expand_dims(eeg, 0)
            env1 = np.expand_dims(env1, 0)
            env2 = np.expand_dims(env2, 0)

            ynew = int(self.model.predict([eeg, env1, env2])[0, 0] > 0.5)

            attention = self.mapping[ynew]

            print("Attention: {0}\t Hit={1}".format(
                attention, ynew == locus_label))

            # DOA
            audio_seg = self.audio[seg*self.audio_wsize:(seg+1)*self.audio_wsize, :]
            theta, pseudo_spectrum, peaks = doa_esti_MUSIC(audio_seg, self.RIRs)

            # plt.plot(pseudo_spectrum)
            # plt.show()

            DOA_esti = theta[peaks] - 90
            DOA_attention = None

            if 'F' not in self.mapping:
                if attention == 'L':
                    DOA_attention = min(DOA_esti)
                elif attention == 'R':
                    DOA_attention = max(DOA_esti)
            elif 'L' not in self.mapping:
                if attention == 'F':
                    DOA_attention = min(abs(DOA_esti))
                elif attention == 'R':
                    DOA_attention = max(DOA_esti)
            elif 'R' not in self.mapping:
                if attention == 'L':
                    DOA_attention = min(DOA_esti)
                elif attention == 'F':
                    DOA_attention = min(abs(DOA_esti))

            print("DOA: {0}".format(DOA_esti))
            print("DOA_attention: {0}".format(DOA_attention))
            rir_idx = np.argmin(abs(self.RIRs.RIR_angles - DOA_attention))

            speeches = DAS(audio_seg, DOA_attention, self.RIRs)

            # _, speeches = fd_gsc(audio_seg, self.RIRs, rir_idx=rir_idx)
            # speeches = speeches[0]

            speeches = speeches / np.max(np.abs(speeches))

            self.speech.append(speeches)

        self.speech = np.concatenate(self.speech, axis=0)
        wavfile.write("./audio_outputs/speech_est.wav", self.RIRs.sample_rate, self.speech)


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes1 = fig.add_subplot(211)
        self.axes1.title.set_text('Enhanced speech')

        self.axes3 = fig.add_subplot(212, projection="polar")
        self.axes3.title.set_text('DOA estimation')
        self.axes3.set_theta_zero_location("N")
        self.axes3.set_theta_direction(-1)
        self.axes3.set_thetamin(-90)
        self.axes3.set_thetamax(90)

        super(MplCanvas, self).__init__(fig)

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, eeg_module, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # set window size
        self.setGeometry(300, 300, 1500, 900)
        self.setWindowTitle('EEG assisted acoustic zoom')

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.setCentralWidget(self.canvas)

        n_data = 50
        self.xdata = list(range(n_data))
        self.ydata = [random.randint(0, 10) for i in range(n_data)]

        self.plot_timer = QtCore.QTimer()
        self.plot_timer.setInterval(500)
        # self.plot_timer.timeout.connect(self.update_plot)

        # timer for controlling the main loop
        self.sys_timer = QtCore.QTimer()
        self.sys_timer.setInterval(2000)
        self.sys_timer.timeout.connect(self.start)
        self.sys_timer.timeout.connect(self.update_plot)

        self.eeg_module = eeg_module

        # Layouts
        # start button
        self.start_button = QtWidgets.QPushButton("Start", self)
        self.start_button.clicked.connect(self.plot_timer.start)
        self.start_button.clicked.connect(self.sys_timer.start)
        self.start_button.resize(self.start_button.sizeHint())
        self.start_button.move(1050, 40)

        # stop button
        self.stop_button = QtWidgets.QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.plot_timer.stop)
        self.stop_button.clicked.connect(self.sys_timer.stop)
        self.stop_button.resize(self.stop_button.sizeHint())
        self.stop_button.move(1150, 40)

        # quit button
        self.quit_button = QtWidgets.QPushButton("Quit", self)
        self.quit_button.clicked.connect(QtWidgets.qApp.quit)
        self.quit_button.resize(self.quit_button.sizeHint())
        self.quit_button.move(1250, 40)

        # Text display
        self.text_display = QtWidgets.QLabel(self)
        self.text_display.setText("Green color shade: AAD hit")
        self.text_display2 = QtWidgets.QLabel(self)
        self.text_display2.setText("Red color shade: AAD miss")
        # color
        self.text_display.setStyleSheet("QLabel {color : green}")
        self.text_display2.setStyleSheet("QLabel {color : red}")
        # font size
        self.text_display.setFont(QtGui.QFont("Helvetica", 12, QtGui.QFont.Bold))
        self.text_display2.setFont(QtGui.QFont("Helvetica", 12, QtGui.QFont.Bold))
        self.text_display.resize(self.text_display.sizeHint())
        self.text_display2.resize(self.text_display2.sizeHint())
        self.text_display.move(1000, 500)
        self.text_display2.move(1000, 525)

        self._plot_ref = None
        self.update_plot()
        self.show()

    def start(self):
        self.canvas.axes3.cla()
        self.canvas.axes3.title.set_text('DOA estimation')
        self.canvas.axes3.set_theta_zero_location("N")
        self.canvas.axes3.set_theta_direction(-1)
        self.canvas.axes3.set_thetamin(-90)
        self.canvas.axes3.set_thetamax(90)

        done = self.eeg_module.step()
        if(done == 1):
            self.sys_timer.stop()
            self.plot_timer.stop()

    def update_plot(self):
        self.ydata = self.eeg_module.get_speech()
        xdata = np.linspace(0, len(self.ydata) / 16000, len(self.ydata))

        # set x axis range
        self.canvas.axes1.set_xlim(0, 30)
        self.canvas.axes1.set_ylim(-1.5, 1.5)
        self.canvas.axes1.plot(xdata, self.ydata, '#0022AA', label='Separated speech', linewidth=0.5)

        doa_theta = self.eeg_module.DOA_attention * np.pi / 180
        self.canvas.axes3.vlines(doa_theta, 0, 1, color='#880044', linewidth=5, label='DOA Estimation')
        
        shade_color = '#009922' if self.eeg_module.hit else '#CC0000'
        if self.eeg_module.attention == 'F':
            self.canvas.axes3.fill_between(np.linspace(-30*np.pi/180, 30*np.pi/180, 100), 0, 1, color=shade_color, alpha=0.5, label='Attention')
        elif self.eeg_module.attention == 'R':
            self.canvas.axes3.fill_between(np.linspace(0*np.pi/180, 90*np.pi/180, 100), 0, 1, color=shade_color, alpha=0.5, label='Attention')
        elif self.eeg_module.attention == 'L':
            self.canvas.axes3.fill_between(np.linspace(-90*np.pi/180, 0*np.pi/180, 100), 0, 1, color=shade_color, alpha=0.5, label='Attention')
        self.canvas.axes3.legend(loc='best',bbox_to_anchor=(-1, 0.2, 0.8, 0.8))

        self.canvas.draw()

if __name__ == "__main__":
    sub_num = 11
    trial_num = 3
    fd_gsc_eeg = FD_GSC_EEG(sub_num, trial_num)
    fd_gsc_eeg.load_data()

    app = QtWidgets.QApplication(sys.argv)

    main = MainWindow(fd_gsc_eeg)

    app.exec_()
