import numpy as np
import scipy
from scipy.signal import stft, istft
from scipy.io import wavfile

from util_tools import *
from MUSIC import *


def fd_gsc(mic, rir_info, fft_length=1024, stft_length=1024, stft_overlap=512):
    RIR = rir_info.RIR_sources
    num_srcs = rir_info.num_sources

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
    
    # w_fas_phase = np.angle(w_fas)
    # plt.figure()
    # for i in range(rir_info.num_mics):
    #     plt.plot(w_fas_phase[:, i, 0])
    #     plt.plot(w_fas_phase[:, i, 1])
    # plt.show()

    # STFT of mic signal
    win = scipy.signal.windows.hamming(stft_length)
    freq, _, stft_mic = stft(mic, fs=rir_info.sample_rate, window=win,
                             nfft=stft_length, nperseg=stft_length, noverlap=stft_overlap, axis=0)
    
    # permute dimensions
    stft_mic = np.transpose(stft_mic, (0, 2, 1))

    # initialize
    mu = 0.8
    alpha = 1e-9

    speech_det = []

    for src in range(num_srcs):
        w_fas_src = w_fas[:, :, src]
        h_w_src = h_w[:, :, src]

        # output space
        err = np.zeros_like(stft_mic[:, :, 0])
        for freq_bin in range(1, stft_length // 2):
            # find the null space of h_w_src
            B = scipy.linalg.null_space(h_w_src[[freq_bin], :])
            w_fas_w = w_fas_src[[freq_bin], :]
            W_coeffs = np.zeros((B.shape[1], 1))

            for t in range(stft_mic.shape[1]):
                y = stft_mic[[freq_bin], [t], :].T
                noise_ref = np.matmul(B.T, y)
                
                d = np.matmul(w_fas_w, y)
                e = d - np.matmul(W_coeffs.conj().T, noise_ref)
                W_coeffs = W_coeffs + (mu / (np.matmul(noise_ref.conj().T, noise_ref) + alpha)) * noise_ref * e.conj()

                err[freq_bin, t] = e
        # ifft
        time_s, speech_esti = istft(err, fs=rir_info.sample_rate, 
                                    window=win, nfft=stft_length, nperseg=stft_length, 
                                    noverlap=stft_overlap, input_onesided=True)
        
        speech_det.append(speech_esti)
        plt.plot(time_s, speech_esti)

    return speech_det


if __name__ == '__main__':
    # load rir data
    rir_info = RIR_Info("Computed_RIRs.mat")

    speech_files = ["speech1.wav", "speech2.wav"]
    noise_files = ["Babble_noise1.wav"]

    plt.figure()

    # generate microphone data
    mic_rec, speech_rec, noise_rec = create_micsigs(
        rir_info, duration=15,
        speeches=speech_files, noise=noise_files, additive_noise=False
    )

    theta, pseudo_spectrum, peaks = doa_esti_MUSIC(mic_rec, rir_info)

    DOA_esti = theta[peaks]
    print("DOAs: ", DOA_esti)

    from scipy.io import savemat
    savemat("../week4/DOA_est.mat", {"DOA_est": DOA_esti})

    plt.subplot(2, 1, 2)
    speeches = fd_gsc(mic_rec, rir_info)

    # normalize the output
    for i in range(len(speeches)):
        speeches[i] = speeches[i] / np.max(np.abs(speeches[i]))
    
    mic_rec[:, 0] = mic_rec[:, 0] / np.max(np.abs(mic_rec[:, 0]))
    wavfile.write("./audio_outputs/speech_rec.wav", rir_info.sample_rate, mic_rec[:, 0])
    wavfile.write("./audio_outputs/speech_est_1.wav", rir_info.sample_rate, speeches[0])
    wavfile.write("./audio_outputs/speech_est_2.wav", rir_info.sample_rate, speeches[1])

    plt.show()