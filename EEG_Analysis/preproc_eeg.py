import psutil
import glob
import numpy as np
import scipy.io as sio
from scipy import signal
from scipy.signal import resample, butter

import concurrent.futures
import pickle


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
    eeg_data_linreg = {}
    eeg_data_cnn = {}

    for idx, matfile in enumerate(eeg_filenames):
        mat_struct = sio.loadmat(matfile)
        trial_name = "_".join(matfile.split("/")[-1].split(".")[0].split("-"))

        mat_struct_linreg = {
            "RawData": None,
            "SampleRate": None,
            "SNR": None,
            "AttendedTrack": {"Envelope": None,
                              "Locus": None,
                              "Gender": None},
            "UnattendedTrack": {"Envelope": None,
                                "Locus": None,
                                "Gender": None},
        }

        mat_struct_cnn = {
            "RawData": None,
            "SampleRate": None,
            "SNR": None,
            "AttendedTrack": {"Envelope": None,
                              "Locus": None,
                              "Gender": None},
            "UnattendedTrack": {"Envelope": None,
                                "Locus": None,
                                "Gender": None},
        }

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

        mat_struct_linreg["RawData"]= eeg_data_linreg_bp
        mat_struct_cnn["RawData"]= eeg_data_cnn_bp
        
        # Filling other fields linreg
        mat_struct_linreg["SampleRate"] = eeg_fs
        mat_struct_linreg["SNR"] = mat_struct["trial"]["FileHeader"][0, 0]["SNR"][0, 0][0, 0]
        mat_struct_linreg["AttendedTrack"]["Envelope"] = mat_struct["trial"]["AttendedTrack"][0, 0]["Envelope"][0, 0][0][:-4]
        mat_struct_linreg["AttendedTrack"]["Locus"] = mat_struct["trial"]["AttendedTrack"][0, 0]["Locus"][0, 0][0]
        mat_struct_linreg["AttendedTrack"]["Gender"] = mat_struct["trial"]["AttendedTrack"][0, 0]["SexOfSpeaker"][0, 0][0]
        mat_struct_linreg["UnattendedTrack"]["Envelope"] = mat_struct["trial"]["UnattendedTrack"][0, 0]["Envelope"][0, 0][0][:-4]
        mat_struct_linreg["UnattendedTrack"]["Locus"] = mat_struct["trial"]["UnattendedTrack"][0, 0]["Locus"][0, 0][0]
        mat_struct_linreg["UnattendedTrack"]["Gender"] = mat_struct["trial"]["UnattendedTrack"][0, 0]["SexOfSpeaker"][0, 0][0]

        # Filling other fields: cnn
        mat_struct_cnn["SampleRate"] = eeg_fs
        mat_struct_cnn["SNR"] = mat_struct["trial"]["FileHeader"][0, 0]["SNR"][0, 0][0, 0]
        mat_struct_cnn["AttendedTrack"]["Envelope"] = mat_struct["trial"]["AttendedTrack"][0, 0]["Envelope"][0, 0][0][:-4]
        mat_struct_cnn["AttendedTrack"]["Locus"] = mat_struct["trial"]["AttendedTrack"][0, 0]["Locus"][0, 0][0]
        mat_struct_cnn["AttendedTrack"]["Gender"] = mat_struct["trial"]["AttendedTrack"][0, 0]["SexOfSpeaker"][0, 0][0]
        mat_struct_cnn["UnattendedTrack"]["Envelope"] = mat_struct["trial"]["UnattendedTrack"][0, 0]["Envelope"][0, 0][0][:-4]
        mat_struct_cnn["UnattendedTrack"]["Locus"] = mat_struct["trial"]["UnattendedTrack"][0, 0]["Locus"][0, 0][0]
        mat_struct_cnn["UnattendedTrack"]["Gender"] = mat_struct["trial"]["UnattendedTrack"][0, 0]["SexOfSpeaker"][0, 0][0]
        
        eeg_data_linreg[trial_name] = mat_struct_linreg
        eeg_data_cnn[trial_name] = mat_struct_cnn

    return {"linreg": eeg_data_linreg, "cnn": eeg_data_cnn}


if __name__ == "__main__":
    # Create dictionary for EEG data
    eeg_data = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=psutil.cpu_count() // 2) as executor:
        future_to_eeg_data = {
            executor.submit(load_and_proc_eeg, subject_id): subject_id
            for subject_id in range(1, 38)
        }

        for future in concurrent.futures.as_completed(future_to_eeg_data):
            subject_id = future_to_eeg_data[future]
            try:
                data = future.result()
                eeg_data["S{:0>3d}".format(subject_id)] = data
            except Exception as exc:
                print(
                    "Subject {:0>3d} generated an exception:".format(subject_id))
                print(exc)

    # Save EEG data
    print("Saving EEG data to disk...")
    with open("eeg_data.pkl", "wb") as f:
        pickle.dump(eeg_data, f)
    print("Done!")
