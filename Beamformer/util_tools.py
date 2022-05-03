import numpy as np
import scipy.io as sio
from scipy.io import wavfile
from scipy.signal import resample, fftconvolve

import matplotlib.pyplot as plt


class RIR_Info():
    def __init__(self, file_name):
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
        
        plt.subplot(2, 1, 1)
        plt.plot(data[:, 0])

        # resample audio data to match the sample rate of the RIR
        if audio_fs != Fs:
            data = resample(data, len(data) // audio_fs * Fs)
        
        # filter the audio data with the RIR per channel
        for j in range(num_mics):
            speech_rec[:, j] += fftconvolve(data[:, j], RIR.RIR_sources[:, j, i], mode="same")

    if additive_noise:
        # find active segments in the speech
        active_segments = np.where(speech_rec[:, 0] > 1e-3 * np.std(speech_rec[:, 0]))
        speech_power = np.var(speech_rec[active_segments, 0])
        add_noise = np.random.normal(0, np.sqrt(0.1 * speech_power), (signal_length, num_mics))
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


if __name__ == '__main__':
    # load rir data
    room_ir_info = RIR_Info("Computed_RIRs.mat")

    speech_files = ["part1_track1_dry.wav", "part1_track2_dry.wav"]
    noise_files = []

    mic_rec, speech_rec, noise_rec = create_micsigs(
        room_ir_info, 10, speeches=speech_files, noise=noise_files, additive_noise=True)
