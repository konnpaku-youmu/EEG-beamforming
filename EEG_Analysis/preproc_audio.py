import psutil
import glob
import numpy as np
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor

import soundfile
import scipy.io as sio
from scipy.signal import resample, fftconvolve


def load_audio_filenames(path):
    """
    Loads audio filenames from a path.
    """
    print("Loading audio filenames...")
    filenames = glob.glob(path+"/*.wav")
    filenames.sort()
    print('Found {} wav files'.format(len(filenames)))
    return filenames


def load_preproc_const():
    audio_preproc_cnst = sio.loadmat("audioprepro_constants.mat")
    g = audio_preproc_cnst['g']
    H = []
    for i in range(g.shape[0]):
        h = g[i, 0][0, 0]['h']
        H.append(h)
    return H


def load_audio_and_proc(fs_r, fs_linreg, fs_cnn, H, n_subbands, wav_file):
    print("Processing {}".format(wav_file))
    # Load audio data
    audio, fs = soundfile.read(wav_file)

    # Resample audio data
    resample_audio = resample(audio, len(audio) // fs * fs_r)

    # Decomposition to subbands
    audio_filt = np.ndarray(shape=(n_subbands, len(resample_audio)))

    for sub_band in range(n_subbands):
        h = H[sub_band][:, 0]

        # Filter audio data
        subband_filt = fftconvolve(resample_audio, h, mode='same')
        # Power compression
        subband_filt = np.power(np.abs(subband_filt), 0.6)

        audio_filt[sub_band, :] = subband_filt

        # Downsample filtered signal to 20Hz & 70Hz
    audio_linreg = resample(
        audio_filt, audio_filt.shape[1] // fs_r * fs_linreg, axis=1)
    audio_cnn = resample(
        audio_filt, audio_filt.shape[1] // fs_r * fs_cnn, axis=1)

    # Summing up subbands
    audio_linreg = np.sum(audio_linreg, axis=0)
    audio_cnn = np.sum(audio_cnn, axis=0)


if __name__ == '__main__':
    # Load audio filenames
    wav_files = load_audio_filenames("./dataset/Audio")

    # Set preprocessing constants
    fs_r = 8000
    fs_linreg = 20
    fs_cnn = 70

    H = load_preproc_const()
    n_subbands = len(H)

    with ProcessPoolExecutor(max_workers=psutil.cpu_count() // 2) as executor:
        for wav_file in wav_files:
            executor.submit(load_audio_and_proc, fs_r, fs_linreg,
                            fs_cnn, H, n_subbands, wav_file)
