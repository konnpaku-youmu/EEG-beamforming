import numpy as np
import tensorflow as tf
import keras

class EEG_classifier:
    def __init__(self):
        self.model = None
        self.env_label = None
        self.eeg_data = None

    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)
        self.model.summary()
    
    def load_data(self, trial_data):
        self.eeg_data = trial_data['eeg'][:350, :]
        print(self.eeg_data.shape)
        trial_data.pop('eeg')

        self.env_label = list(trial_data.keys())
        self.envelopes = list(trial_data.values())
        self.envelopes = [ env[:350] for env in self.envelopes]
    
    def predict(self):
        return self.model.predict([self.eeg_data, self.envelopes[0], self.envelopes[1]])
