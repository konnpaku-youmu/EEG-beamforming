import glob
import os
import numpy as np
import scipy.io as sio
import mat73
from scipy.io import wavfile
from scipy.signal import resample, fftconvolve

import matplotlib.pyplot as plt


class RIR_Info():
    def __init__(self, file_name=None):
        if file_name is not None:
            self.load_from_mat(file_name)
        else:
            self.sample_rate = None
            self.mic_pos = None
            self.reverb_time = None
            self.RIR_noise = None
            self.RIR_sources = None
            self.room_dimension = None
            self.source_pos = None
            self.noise_pos = None
            self.num_mics = None
            self.num_sources = None

    def load_from_mat(self, file_name):
        mat_data = sio.loadmat(file_name)
        self.sample_rate = mat_data['fs_RIR'][0, 0]
        self.mic_pos = mat_data['m_pos']
        self.reverb_time = mat_data['rev_time'][0, 0]
        self.RIR_noise = mat_data['RIR_noise']
        self.RIR_sources = mat_data['RIR_sources']
        self.room_dimension = mat_data['room_dim'][0]
        self.source_pos = mat_data['s_pos']
        self.noise_pos = mat_data['v_pos']
        self.num_mics = self.mic_pos.shape[0]
        self.num_sources = self.source_pos.shape[0]

    def mini_load(self, fs, rir, num_mics):
        self.sample_rate = fs
        self.RIR_sources = rir
        self.num_mics = num_mics

    def get_intermic_dist(self):
        # calculate inter-microphone distance
        return np.array(list(map(lambda x: np.linalg.norm(x - self.mic_pos[0]), self.mic_pos)), dtype=np.float32, ndmin=2)

    def get_reverb_time(self):
        return self.reverb_time

    def __str__(self) -> str:
        # print all member variables
        return "sample_rate:\t{}\nmic_pos:\t{}\nreverb_time:\t{}\nRIR_noise:\t{}\nRIR_sources:\t{}\nroom_dimension:\t{}\nsource_pos:\t{}\nnoise_pos:\t{}".format(
            self.sample_rate, self.mic_pos.shape, self.reverb_time, self.RIR_noise.shape, self.RIR_sources.shape, self.room_dimension, self.source_pos.shape, self.noise_pos.shape)


def create_micsigs(RIR, duration, speeches=None, noise=None, additive_noise=False):

    print('Creating micsig...')

    try:
        assert speeches is not None
    except AssertionError:
        print('No speech files provided. Aborting...')
        return

    Fs = RIR.sample_rate
    num_mics = RIR.mic_pos.shape[0]
    signal_length = Fs * duration

    speech_rec = np.zeros((signal_length, num_mics), dtype=np.float32)
    noise_rec = np.zeros((signal_length, num_mics), dtype=np.float32)

    for i in range(len(speeches)):
        audio_fs, data = wavfile.read("./audio_files/" + speeches[i])
        # normalize int16 to float32
        data = data.astype(np.float32) / np.iinfo(np.int16).max

        data = np.repeat(data[:duration * audio_fs], num_mics, axis=0)
        data = data.reshape((signal_length, num_mics))

        # resample audio data to match the sample rate of the RIR
        if audio_fs != Fs:
            data = resample(data, len(data) // audio_fs * Fs)

        # filter the audio data with the RIR per channel
        for j in range(num_mics):
            speech_rec[:, j] += fftconvolve(data[:, j],
                                            RIR.RIR_sources[:, j, i], mode="same")

    if additive_noise:
        # find active segments in the speech
        active_segments = np.where(
            speech_rec[:, 0] > 1e-3 * np.std(speech_rec[:, 0]))
        speech_power = np.var(speech_rec[active_segments, 0])
        add_noise = np.random.normal(0, np.sqrt(
            0.1 * speech_power), (signal_length, num_mics))
        noise_rec += add_noise

    for i in range(len(noise)):
        audio_fs, data = wavfile.read("./audio_files/" + noise[i])
        # normalize int16 data to float 0-1
        data = data.astype(np.float32) / np.iinfo(np.int16).max
        data = np.repeat(data[:duration * audio_fs], num_mics, axis=0)
        data = data.reshape((signal_length, num_mics))

        # resample audio data to match the sample rate of the RIR
        if audio_fs != Fs:
            data = resample(data, len(data) // audio_fs * Fs)

        noise_rec += fftconvolve(data, RIR.RIR_noise, mode="same")

    mic_rec = speech_rec + noise_rec

    return mic_rec, speech_rec, noise_rec


def load_timevar_angles():
    with open("./Data/Sequence1_anechoic/timeIntervals.txt", 'r') as f:
        time_intervals = f.readlines()
    time_intervals = [float(x.strip()) for x in time_intervals]
    # accumulate time intervals
    timestamps = np.cumsum(time_intervals)

    with open("./Data/Sequence1_anechoic/anglesL.txt", 'r') as f:
        angles_l = f.readlines()
    angles_l = ['s' + x.strip().split('.')[0] for x in angles_l]

    with open("./Data/Sequence1_anechoic/anglesR.txt", 'r') as f:
        angles_r = f.readlines()
    angles_r = ['s' + x.strip().split('.')[0] for x in angles_r]

    return list(zip(timestamps, angles_l, angles_r))


def load_RIRs():
    path = "./Data/IR_LMA_5cm_conv2/convention2/"
    angles = os.listdir(path)

    RIR = {}
    for angle in angles:
        files = glob.glob(path + angle + "/*.wav")

        RIR_multchannel = None
        for file in files:
            fs, audio_rir = wavfile.read(file)
            # normalize to float 0-1 according to data type
            audio_rir = audio_rir.astype(np.float32) / np.iinfo(audio_rir.dtype).max

            # concatenate all channels
            if RIR_multchannel is None:
                RIR_multchannel = audio_rir
            else:
                RIR_multchannel = np.concatenate((RIR_multchannel, audio_rir))

        # reshape to (num_channels, num_samples)
        RIR_multchannel = RIR_multchannel.reshape(
            (len(files), RIR_multchannel.shape[0] // len(files)))

        RIR[angle] = RIR_multchannel.T

    return fs, RIR


def load_audios():
    path = "./Data/Sequence1_anechoic/"
    audio_files = glob.glob(path + "y_LMA_M*.wav")
    audio_files.sort()

    recordings = None

    for file in audio_files:
        fs, audio = wavfile.read(file)
        # normalize to float -1-1 according to data type
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max

        # concatenate all channels
        if recordings is None:
            recordings = audio
        else:
            recordings = np.concatenate((recordings, audio))

    # reshape to (num_channels, num_samples)
    recordings = recordings.reshape(
        (len(audio_files), recordings.shape[0] // len(audio_files)))

    # load reference audio
    fs, ref_audio_l = wavfile.read(path + "sL_LMA.wav")
    fs, ref_audio_r = wavfile.read(path + "sR_LMA.wav")

    return fs, recordings.T, ref_audio_l, ref_audio_r

def load_eeg_data(file_name):
    data = mat73.loadmat(file_name,  use_attrdict=True)
    subject_id = ["{:03d}".format(i) for i in range(1, 38) if i != 22]
    subject_data = data['Data']

    eeg_dict = dict(zip(subject_id, subject_data))
    
    return eeg_dict
