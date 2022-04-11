import glob
import numpy as np
import pickle

import matplotlib.pyplot as plt

class Trial:
    """
    Class for EEG data
    """
    def __init__(self):
        self.sample_rate = None
        self.SNR = None
        self.att_track_name = None
        self.attended_track = None
        self.unatt_track_name = None
        self.unattended_track = None
        self.eeg_data = None
        self.audio_data = None
        return
    
    def load(self, trial, audio_data):
        """
        Loads EEG data from a mat file.
        """
        self.sample_rate = trial["SampleRate"]
        self.SNR = trial["SNR"]
        self.eeg_data = trial["RawData"]
        self.att_track_name = trial["AttendedTrack"]["Envelope"]
        self.attended_track = audio_data[self.att_track_name]["linreg"]
        self.unatt_track_name = trial["UnattendedTrack"]["Envelope"]
        self.unattended_track = audio_data[self.unatt_track_name]["linreg"]
        return


class EEG_data:
    def __init__(self, subject_id):
        self.subject_id = subject_id
        self.trials = {}
        return

    def parse_data(self, eeg_data, audio_data):
        """
        Parses EEG data and audio data.
        """
        __trials_org__ = eeg_data[self.subject_id]["linreg"]
        for trial_name in __trials_org__.keys():
            trial = Trial()
            trial.load(__trials_org__[trial_name], audio_data)
            self.trials[trial_name] = trial
        
        return


def load_data(path):
    """
    Loads audio and eeg data
    """
    print("Loading audio and eeg data...")
    # Load eeg data
    eeg_file = open(path + "/eeg_data.pkl", "rb")
    eeg_data = pickle.load(eeg_file)
    eeg_file.close()
    # Load audio data
    audio_file = open(path + "/audio_data.pkl", "rb")
    audio_data = pickle.load(audio_file)
    audio_file.close()
    
    print("Done.")
    return audio_data, eeg_data

def create_lagmat(N_delay, eeg_concat, train_len):
    M = []
    
    for col in eeg_concat.transpose():
        channel_i = np.zeros((train_len, N_delay))
        for lag in range(N_delay):
            channel_i[:train_len-lag, lag] = col[lag:train_len]
        M.append(channel_i)

    return np.concatenate(M, axis=1)


if __name__ == "__main__":
    path = "."
    audio_dataset, eeg_dataset = load_data(path)

    subject_list = ["S"+folder.split('/')[-1] for folder in glob.glob("./dataset/EEG/*")]

    # Parse data
    eeg_data = []
    for subject in subject_list:
        print("Parsing data for subject: " + subject)
        eeg_data_per_subj = EEG_data(subject)
        eeg_data_per_subj.parse_data(eeg_dataset, audio_dataset)
        eeg_data.append(eeg_data_per_subj)
    
    
    # Calculate the attended envelope
    N_delay = 6
    test_len = 1000
    N_subj = len(eeg_data)

    for subject in eeg_data:
        eeg_concat = None
        att_env_concat = None
        unatt_env_concat = None

        eeg_concat = np.concatenate(
            [trial.eeg_data for trial in subject.trials.values()],
            axis=0
        )
        att_env_concat = np.concatenate(
            [trial.attended_track for trial in subject.trials.values()],
            axis=0
        )

        unatt_env_concat = np.concatenate(
            [trial.unattended_track for trial in subject.trials.values()],
            axis=0
        )

        # Calculated the time-lagged eeg data per channel
        train_len = eeg_concat.shape[0] - test_len

        M = create_lagmat(N_delay, eeg_concat, train_len)
        
        R = M.transpose().dot(M)
        r = M.transpose().dot(att_env_concat[:train_len])
        d = np.linalg.solve(R, r)

        # test
        test_eeg = eeg_concat[7*test_len:8*test_len, :]
        envelope_test = att_env_concat[7*test_len:8*test_len]
        unatt_envelope = unatt_env_concat[7*test_len:8*test_len]

        M_test = create_lagmat(N_delay, test_eeg, test_len)
        envelope_esti = M_test.dot(d)
        
        # calculate the correlation between the estimated envelope and the unattended envelope and the test envelope
        corr_att = np.corrcoef(envelope_esti, unatt_envelope)[0, 1]
        corr_unatt = np.corrcoef(envelope_esti, envelope_test)[0, 1]
        print(corr_att > corr_unatt)
