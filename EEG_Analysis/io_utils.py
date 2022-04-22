import pickle

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
    
    def load(self, trial, audio_data, audio_mode):
        """
        Loads EEG data from a mat file.
        """
        self.sample_rate = trial["SampleRate"]
        self.SNR = trial["SNR"]
        self.eeg_data = trial["RawData"]
        self.att_track_name = trial["AttendedTrack"]["Envelope"]
        self.attended_track = audio_data[self.att_track_name][audio_mode]
        self.unatt_track_name = trial["UnattendedTrack"]["Envelope"]
        self.unattended_track = audio_data[self.unatt_track_name][audio_mode]
        return


class EEG_data:
    def __init__(self, subject_id, mode):
        self.subject_id = subject_id
        self.mode = mode
        self.trials = {}
        return

    def parse_data(self, eeg_data, audio_data):
        """
        Parses EEG data and audio data.
        """
        __trials_org__ = eeg_data[self.subject_id][self.mode]
        for trial_name in __trials_org__.keys():
            trial = Trial()
            trial.load(__trials_org__[trial_name], audio_data, self.mode)
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