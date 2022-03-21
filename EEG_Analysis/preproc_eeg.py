import psutil
import glob
import numpy as np
import scipy.io as sio
from scipy import signal
from scipy.signal import resample, butter

from concurrent.futures import ProcessPoolExecutor


def load_eeg_filenames(path):
    """
    Loads EEG filenames from a path.
    """
    filenames = glob.glob(path + "/*.mat")
    print("Found {0} mat files in {1}".format(len(filenames), path))
    return filenames


def load_and_proc_eeg(subject_id, fs_linreg=20, fs_cnn=70):
    """
    Loads EEG data from a mat file.
    """
    print("Processing subject {:0>3d}".format(subject_id))
    eeg_filenames = load_eeg_filenames(
        "./dataset/EEG/{:0>3d}".format(subject_id))
    eeg_data_linreg = np.ndarray(shape=(len(eeg_filenames), 1000, 64))
    eeg_data_cnn = np.ndarray(shape=(len(eeg_filenames), 3500, 64))

    for idx, matfile in enumerate(eeg_filenames):
        mat_struct = sio.loadmat(matfile)

        eeg_data = mat_struct["trial"]["RawData"][0, 0]["EegData"][0, 0]
        eeg_fs = mat_struct["trial"]["FileHeader"][0,
                                                   0]["SampleRate"][0, 0][0, 0]

        # Resample EEG data
        eeg_raw_linreg = resample(
            eeg_data, eeg_data.shape[0] // eeg_fs * fs_linreg, axis=0
        )
        eeg_raw_cnn = resample(
            eeg_data, eeg_data.shape[0] // eeg_fs * fs_cnn, axis=0)

        # Bandpass filter EEG data: 4-th order butterworth filter
        sos_linreg = butter(
            4, [1, 9], btype="bandpass", fs=fs_linreg, analog=False, output="sos"
        )
        sos_cnn = butter(
            4, [1, 32], btype="bandpass", fs=fs_cnn, analog=False, output="sos"
        )
        eeg_data_linreg_bp = signal.sosfilt(sos_linreg, eeg_raw_linreg, axis=0)
        eeg_data_cnn_bp = signal.sosfilt(sos_cnn, eeg_raw_cnn, axis=0)

        eeg_data_linreg[idx, :, :] = eeg_data_linreg_bp
        eeg_data_cnn[idx, :, :] = eeg_data_cnn_bp

    return eeg_data_linreg, eeg_data_cnn


if __name__ == "__main__":
    # Create dictionary for EEG data
    eeg_data = {}
    for subject_id in range(1, 38):
        # Load and process EEG data
        eeg_data_linreg, eeg_data_cnn = load_and_proc_eeg(subject_id)
        eeg_data["S{:0>3d}".format(subject_id)] = {
            "linreg": eeg_data_linreg,
            "cnn": eeg_data_cnn,
        }

    # Save EEG data
    print("Saving EEG data to disk...")
    sio.savemat("eeg_data.mat", eeg_data)
