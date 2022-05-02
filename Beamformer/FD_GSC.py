from util_tools import *
from MUSIC import *

import numpy as np
from scipy.fft import fft


def fd_gsc(mic, rir_info, fft_length=1024, stft_length=1024, stft_overlap=50):
    RIR = rir_info.RIR_sources
    num_srcs = rir_info.num_sources

    h_w = np.fft.fft(RIR, fft_length, axis=0)

    # normalize per source
    for i in range(num_srcs):
        h_w[:, :, i] = h_w[:, :, i] / h_w[:, 0:1, i]

    # STFT of mic signal
    win = scipy.signal.windows.hamming(300)
    freq, t, stft_mic = stft(mic, fs=rir_info.sample_rate, window=win,
                             nfft=stft_length, nperseg=300, noverlap=stft_overlap, axis=0)
    
    # plot stft_mic
    print(stft_mic.shape)


if __name__ == '__main__':
    # load rir data
    rir_info = RIR_Info("Computed_RIRs.mat")

    speech_files = ["part1_track1_dry.wav", "part1_track2_dry.wav"]
    noise_files = []

    # generate microphone data
    mic_rec, speech_rec, noise_rec = create_micsigs(
        rir_info, duration=10,
        speeches=speech_files, noise=noise_files, additive_noise=True
    )

    theta, pseudo_spectrum, peaks = doa_esti_MUSIC(mic_rec, rir_info)

    DOA_esti = theta[peaks]

    # from scipy.io import savemat
    # savemat("../week4/DOA_est.mat", {"DOA_est": DOA_esti})

    # # Plot the pseudo spectrum: mag to angle
    # plt.plot(theta, pseudo_spectrum)
    # plt.plot(theta[peaks], pseudo_spectrum[peaks] , 'rx')
    # plt.xlabel("Angle")
    # plt.ylabel("Magnitude")
    # plt.title("Pseudo Spectrum")
    # plt.show()

    fd_gsc(mic_rec, rir_info)
