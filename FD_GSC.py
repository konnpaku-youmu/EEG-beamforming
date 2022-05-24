from matplotlib.pyplot import step
import numpy as np
import scipy
from scipy.fft import fft
from scipy.signal import stft, istft
from scipy.io import wavfile
import matplotlib.pyplot as plt

from util_tools import *
from MUSIC import *


class FD_GSC:
    def __init__(self, rir_info, win_len, fft_len):
        self.rir_info = rir_info
        self.audio_win_len = win_len
        self.audio_data = np.zeros(
            (self.audio_win_len, self.rir_info.num_mics))
        self.fft_len = fft_len
        self.filtered_data = np.zeros(
            (self.audio_win_len, self.rir_info.num_mics))
        self.W_coeffs = None

        self.init_cnt = 0

    def initialize():
        pass

    def update_rir(self, rir):
        self.rir_info.RIR_sources = rir

    def update_sample(self, sample):
        if self.init_cnt < self.audio_win_len:
            self.audio_data[self.init_cnt] = sample
            self.init_cnt += 1
        else:
            self.audio_data = np.roll(self.audio_data, -1, axis=0)
            self.audio_data[-1] = sample

    def step(self):
        self.W_coeffs, self.filtered_data = fd_gsc(
            self.audio_data, self.rir_info, self.fft_len, self.fft_len,
            self.fft_len//2, self.W_coeffs, self.init_cnt
        )
        wavfile.write("./audio_outputs/speech_est_1.wav", self.rir_info.sample_rate,
                      self.filtered_data[0] / np.max(self.filtered_data[0]))

    def plot_all(self):
        # print(self.filtered_data.shape)
        plt.plot(self.audio_data[:, 0])
        plt.plot(self.filtered_data[0])


def fd_gsc(mic, rir_info, fft_length=512, stft_length=512, stft_overlap=256, init_w=None, iter_n=None, rir_idx=None):
    RIR = rir_info.RIR_sources
    
    if rir_idx is not None:
        RIR = RIR[:, :, [rir_idx]]
    
    num_srcs = RIR.shape[-1]

    h_w = np.fft.fft(RIR, fft_length, axis=0)

    # normalize per source
    for i in range(num_srcs):
        h_w[:, :, i] = h_w[:, :, i] / h_w[:, 0:1, i]

    h_w = h_w[:fft_length // 2 + 1, :, :]

    # define the filter-and-sum beamformer
    w_fas = np.zeros_like(h_w)
    for i in range(num_srcs):
        for freq_bin in range(fft_length // 2 + 1):
            w_fas[freq_bin, :, i] = h_w[freq_bin, :, i] / np.matmul(h_w[freq_bin, :, i].conj().T, h_w[freq_bin, :, i])

    # STFT of mic signal
    win = scipy.signal.windows.hamming(stft_length)
    freq, _, stft_mic = stft(mic, fs=rir_info.sample_rate, window=win,
                             nfft=stft_length, nperseg=stft_length, noverlap=stft_overlap, axis=0)

    # permute dimensions
    stft_mic = np.transpose(stft_mic, (0, 2, 1))

    # initialize
    mu = 1.5
    alpha = 1e-8

    speech_det = []

    W_coeffs = None
    if init_w is None:
        W_coeffs = np.zeros((4, stft_length // 2, num_srcs),
                            dtype=np.complex128)
    elif init_w is not None and iter_n == 0:
        W_coeffs = np.zeros((4, stft_length // 2, num_srcs),
                            dtype=np.complex128)
    else:
        W_coeffs = init_w

    for src in range(num_srcs):
        w_fas_src = w_fas[:, :, src]
        h_w_src = h_w[:, :, src]

        # output space
        err = np.zeros_like(stft_mic[:, :, 0])
        for freq_bin in range(1, stft_length // 2):
            # find the null space of h_w_src
            B = scipy.linalg.null_space(h_w_src[[freq_bin], :])
            w_fas_w = w_fas_src[[freq_bin], :]
            # W_coeffs = np.zeros((B.shape[1], 1))

            for t in range(stft_mic.shape[1]):
                y = stft_mic[[freq_bin], [t], :].T
                noise_ref = np.matmul(B.T, y)

                d = np.matmul(w_fas_w, y)
                e = d - np.matmul(W_coeffs[:, [freq_bin], [src]].conj().T, noise_ref)
                W_coeffs[:, [freq_bin], [src]] = W_coeffs[:, [freq_bin], [src]] + (mu / (np.matmul(noise_ref.conj().T, noise_ref) + alpha)) * noise_ref * e.conj()

                err[freq_bin, t] = e
        # ifft
        time_s, speech_esti = istft(err, fs=rir_info.sample_rate,
                                    window=win, nfft=stft_length, nperseg=stft_length,
                                    noverlap=stft_overlap, input_onesided=True)

        speech_det.append(speech_esti)

    return W_coeffs, speech_det


if __name__ == '__main__':
    # load rir data
    rir_info = RIR_Info()
    rir_info.load_from_mat_legacy("Computed_RIRs.mat")

    speech_files = ["speech1.wav", "speech2.wav"]
    noise_files = ["Babble_noise1.wav"]

    # generate microphone data
    mic_rec, speech_rec, noise_rec = create_micsigs(
        rir_info, duration=10,
        speeches=speech_files, noise=noise_files, additive_noise=True
    )

    theta, pseudo_spectrum, peaks = doa_esti_MUSIC(mic_rec, rir_info)

    DOA_esti = theta[peaks]

    _, speeches = fd_gsc(mic_rec, rir_info)

    # normalize the output
    for i in range(len(speeches)):
        speeches[i] = speeches[i] / np.max(np.abs(speeches[i]))

    mic_rec[:, 0] = mic_rec[:, 0] / np.max(np.abs(mic_rec[:, 0]))
    wavfile.write("./audio_outputs/speech_rec.wav",
                  rir_info.sample_rate, mic_rec[:, 0])
    wavfile.write("./audio_outputs/speech_est_1.wav",
                  rir_info.sample_rate, speeches[0])
    wavfile.write("./audio_outputs/speech_est_2.wav",
                  rir_info.sample_rate, speeches[1])
