import glob
import numpy as np

import matplotlib.pyplot as plt
from io_utils import *


def create_lagmat(N_delay, eeg_concat):
    M = []
    for col in eeg_concat.transpose():
        channel_i = np.zeros((eeg_concat.shape[0], N_delay))
        for lag in range(N_delay):
            channel_i[:eeg_concat.shape[0]-lag, lag] = col[lag:]
        M.append(channel_i)

    return np.concatenate(M, axis=1)


if __name__ == "__main__":
    path = "."
    audio_dataset, eeg_dataset = load_data(path)

    subject_list = ["S"+folder.split('/')[-1] for folder in glob.glob("./dataset/EEG/*")]

    # Parse data
    eeg_data = []
    for subject in subject_list:
        eeg_data_per_subj = EEG_data(subject, "linreg")
        eeg_data_per_subj.parse_data(eeg_dataset, audio_dataset)
        eeg_data.append(eeg_data_per_subj)
    
    # Calculate the attended envelope
    N_delay = 6
    trial_len = list(eeg_data[0].trials.values())[0].eeg_data.shape[0]
    N_subj = len(eeg_data)
    
    accuracy = []
    
    for test_len in [50, 200, 1000]:
        
        accuracy_per_len = []
        
        for subject in eeg_data:
            
            print("Calculating for subject: " + subject.subject_id)

            N_trials = len(subject.trials)

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

            ccnt = 0
            for i, trial in enumerate(subject.trials.values()):
                # cross validation: exclude one trial at a time
                idx_valid = [idx for idx in range(eeg_concat.shape[0]) if idx not in range(i*trial_len, (i+1)*trial_len)]
                eeg_train = eeg_concat[idx_valid, :]
                att_env_train = att_env_concat[idx_valid]
                unatt_env_train = unatt_env_concat[idx_valid]
                
                # Calculated the time-lagged eeg data per channel
                M = create_lagmat(N_delay, eeg_train)
                R = M.transpose().dot(M)
                r = M.transpose().dot(att_env_train)
                d = np.linalg.solve(R, r)

                # Test against each trial
                for j in range(N_trials):
                    eeg_test = eeg_concat[j*trial_len:j*trial_len+test_len, :]
                    att_env_test = att_env_concat[j*trial_len:j*trial_len+test_len]
                    unatt_env_test = unatt_env_concat[j*trial_len:j*trial_len+test_len]

                    # Calculate the time-lagged eeg data per channel
                    M_test = create_lagmat(N_delay, eeg_test)
                    env_esti = M_test.dot(d)

                    # Calculate correlations: attended and unattended
                    corr_att = np.corrcoef(env_esti, att_env_test)[0, 1]
                    corr_unatt = np.corrcoef(env_esti, unatt_env_test)[0, 1]

                    if(corr_att > corr_unatt):
                        ccnt += 1
            
            # Accuracy of the subject
            accuracy_per_len.append(ccnt / (N_trials**2))
        
        accuracy.append(accuracy_per_len)
    
    # Plot accuracy in box plot
    plt.figure()
    plt.boxplot(accuracy)
    plt.xticks([1, 2, 3], ["50", "200", "1000"])
    plt.ylabel("Accuracy")
    plt.xlabel("Test length")
    plt.title("Accuracy of the linear model")
    plt.show()
