import psutil
import glob
import numpy as np

import concurrent.futures

import soundfile
import scipy.io as sio
from scipy.signal import resample, fftconvolve, butter, sosfilt


def load_audio_filenames(path):
    """
    Loads audio filenames from a path.
    """
    print("Loading audio filenames...")
    filenames = glob.glob(path + "/*.wav")
    filenames.sort()
    print("Found {} wav files".format(len(filenames)))
    return filenames


def load_preproc_const():
    audio_preproc_cnst = sio.loadmat("audioprepro_constants.mat")
    g = audio_preproc_cnst["g"]
    H = []
    for i in range(g.shape[0]):
        h = g[i, 0][0, 0]["h"]
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
        subband_filt = fftconvolve(resample_audio, h, mode="same")
        # Power compression
        neg_idx = np.where(subband_filt < 0)
        subband_filt = np.power(np.abs(subband_filt), 0.6)
        subband_filt[neg_idx] = -subband_filt[neg_idx]

        audio_filt[sub_band, :] = subband_filt

    # Downsample filtered signal to 20Hz & 70Hz
    audio_linreg = resample(
        audio_filt, audio_filt.shape[1] // fs_r * fs_linreg, axis=1)
    audio_cnn = resample(
        audio_filt, audio_filt.shape[1] // fs_r * fs_cnn, axis=1)

    # Summing up subbands
    audio_linreg = np.sum(audio_linreg, axis=0)
    audio_cnn = np.sum(audio_cnn, axis=0)

    # Bandpass filter: 4-th butterworth filter
    sos_linreg = butter(4, [1, 9], btype="bandpass",
                        fs=fs_linreg, output="sos")

    sos_cnn = butter(4, [1, 32], btype="bandpass", fs=fs_cnn, output="sos")
    audio_linreg_bp = sosfilt(sos_linreg, audio_linreg)
    audio_cnn_bp = sosfilt(sos_cnn, audio_cnn)

    return {"linreg": audio_linreg_bp, "cnn": audio_cnn_bp}


if __name__ == "__main__":
    # Load audio filenames
    wav_files = load_audio_filenames("./dataset/Audio")

    # Set preprocessing constants
    fs_r = 8000
    fs_linreg = 20
    fs_cnn = 70

    H = load_preproc_const()
    n_subbands = len(H)

    audio_data = {}
    # Load audio data and process
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=psutil.cpu_count() // 2
    ) as executor:
        future_to_wav_file = {
            executor.submit(
                load_audio_and_proc, fs_r, fs_linreg, fs_cnn, H, n_subbands, wav_file
            ): wav_file
            for wav_file in wav_files
        }

        for future in concurrent.futures.as_completed(future_to_wav_file):
            wav_file = future_to_wav_file[future]
            try:
                data = future.result()
                audio_data[wav_file.split("/")[-1].split(".")[0]] = data
            except Exception as exc:
                print("{} generated an exception: {}".format(wav_file, exc))

    # Save audio data
    sio.savemat("audio_data.mat", audio_data)
