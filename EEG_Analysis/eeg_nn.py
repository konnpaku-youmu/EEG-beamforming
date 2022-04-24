import glob
import numpy as np
#import torch
import tensorflow as tf
import keras

from io_utils import *

class EEG_NN():
    def __init__(self, window_len):
        self.window_len = window_len
        # inputs: 
        # -- eeg: window_len x 64
        self.eeg = keras.Input(shape=(window_len, 64))
        # -- env1: window_len x 1
        self.env1 = keras.Input(shape=(window_len, 1))
        # -- env2: window_len x 1
        self.env2 = keras.Input(shape=(window_len, 1))

        # Define the model structure
        # -- conv1: 64 x 64 x 32
        self.conv1 = keras.layers.Conv1D(filters=7, kernel_size=3, padding='same', activation='relu')(self.eeg)
        # -- conv2: 64 x 64 x 32
        self.conv2 = keras.layers.Conv1D(filters=15, kernel_size=3, padding='same', activation='relu')(self.conv1)
        # -- FC1: 640 x 1
        self.fc1 = keras.layers.Dense(1, activation='relu')(self.conv2)

        print(self.fc1.shape)
        
        # Cosine similarity with env1 and env2
        # -- cos_sim: 640 x 1
        self.cos_sim1 = keras.layers.Dot(axes=1)([self.fc1, self.env1])
        # -- cos_sim: 640 x 1
        self.cos_sim2 = keras.layers.Dot(axes=1)([self.fc1, self.env2])
        # Concatenate the cosine similarities
        # -- cos_sim: 640 x 2
        self.cos_sim = keras.layers.concatenate([self.cos_sim1, self.cos_sim2])
        self.flatten = keras.layers.Flatten()(self.cos_sim)

        # output: 640x1
        self.output = keras.layers.Dense(1, activation='sigmoid')(self.flatten)

        self.model = keras.Model(inputs=[self.eeg, self.env1, self.env2], outputs=self.output)

        self.model.summary()

# class EEG_NN(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(EEG_NN, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         # Define the layers
#         self.fc1 = torch.nn.Linear(input_size, hidden_size)
#         self.fc2 = torch.nn.Linear(hidden_size, output_size)
#         # cosine similarity
#         self.cos1 = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
#         self.cos2 = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        
#         return

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
    
#     def train(self, train_data, train_labels, epochs, batch_size, learning_rate, criterion, optimizer):
#         # Define the loss function
#         criterion = torch.nn.MSELoss()
#         # Define the optimizer
#         optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
#         # Train the model
#         for epoch in range(epochs):
#             for i in range(0, len(train_data), batch_size):
#                 # Get the batch
#                 batch_data = train_data[i:i+batch_size]
#                 batch_labels = train_labels[i:i+batch_size]
#                 # Forward pass
#                 outputs = self(batch_data)
#                 # Compute loss
#                 loss = criterion(outputs, batch_labels)
#                 # Backward pass
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#         return


def split_train_valid(eeg_joined, att_env_joined, unatt_env_joined):
    # randomly split data into train and validation sets
    # (train, validation)
    train_idx = np.random.choice(
        eeg_joined.shape[0],
        size=int(eeg_joined.shape[0]*0.8),
        replace=False
    )
    valid_idx = np.array(
        list(set(range(eeg_joined.shape[0])) - set(train_idx))
    )
    eeg_train = eeg_joined[train_idx, :, :]
    att_env_train = att_env_joined[train_idx, :]
    unatt_env_train = unatt_env_joined[train_idx, :]

    eeg_valid = eeg_joined[valid_idx, :, :]
    att_env_valid = att_env_joined[valid_idx, :]
    unatt_env_valid = unatt_env_joined[valid_idx, :]

    return (eeg_train, att_env_train, unatt_env_train), (eeg_valid, att_env_valid, unatt_env_valid)


def resize_window(eeg_train, att_env_train, unatt_env_train, window_len, trial_len):
    N_win_per_trial = int(trial_len / window_len)
    # truncate data to make it divisible by window_len
    eeg_train = eeg_train[:, :N_win_per_trial*window_len, :]
    att_env_train = att_env_train[:, :N_win_per_trial*window_len]
    unatt_env_train = unatt_env_train[:, :N_win_per_trial*window_len]
    # resize according to window_len
    eeg_train = eeg_train.reshape(
        (eeg_train.shape[0]*N_win_per_trial, window_len, eeg_train.shape[2])
    )
    att_env_train = att_env_train.reshape(
        (att_env_train.shape[0]*N_win_per_trial, window_len)
    )
    unatt_env_train = unatt_env_train.reshape(
        (unatt_env_train.shape[0]*N_win_per_trial, window_len)
    )

    return eeg_train, att_env_train, unatt_env_train


def batch_equalizer(eeg, env1, env2, labels):
    # present each of the eeg segments twice, where the envelopes, and thus the labels 
    # are swapped around. EEG presented in small segments [bs, window_length, 64]
    return np.concatenate([eeg,eeg], axis=0), np.concatenate([env1, env2], axis=0),np.concatenate([env2, env1], axis=0), np.concatenate([labels, (labels+1)%2], axis=0)
    

if __name__ == "__main__":
    path = "."
    audio_dataset, eeg_dataset = load_data(path)

    subject_list = ["S"+folder.split('/')[-1]
                    for folder in glob.glob("./dataset/EEG/*")]

    # Parse data
    subj_data = []
    for subject in subject_list:
        eeg_data_per_subj = EEG_data(subject, "cnn")
        eeg_data_per_subj.parse_data(eeg_dataset, audio_dataset)
        subj_data.append(eeg_data_per_subj)

    # stack data in 3-d array
    # (trial, signal_length, channel)
    eeg_joined = None
    att_env_joined = None
    unatt_env_joined = None
    eeg_joined = np.stack(
        [trial.eeg_data for subject in subj_data for trial in subject.trials.values()]
    )
    att_env_joined = np.stack(
        [trial.attended_track for subject in subj_data for trial in subject.trials.values()]
    )
    unatt_env_joined = np.stack(
        [trial.unattended_track for subject in subj_data for trial in subject.trials.values()]
    )

    # split data into train and validation sets
    (eeg_train, env1_train, env2_train), (eeg_valid, env1_valid, env2_valid) = split_train_valid(
        eeg_joined, att_env_joined, unatt_env_joined
    )

    window_len = 320
    trial_len = eeg_train.shape[1]
    eeg_train, env1_train, env2_train = resize_window(
        eeg_train, env1_train, env2_train, window_len, trial_len
    )
    eeg_valid, env1_valid, env2_valid = resize_window(
        eeg_valid, env1_valid, env2_valid, window_len, trial_len
    )

    labels_valid = np.zeros((eeg_valid.shape[0],1)).astype(int)
    labels_train = np.zeros((eeg_train.shape[0],1)).astype(int)
    eeg_train, env1_train, env2_train, labels_train= batch_equalizer(eeg_train, env1_train, env2_train, labels_train)
    eeg_valid, env1_valid, env2_valid, labels_valid= batch_equalizer(eeg_valid, env1_valid, env2_valid, labels_valid)

    # print shapes
    print("eeg_train shape:", eeg_train.shape)
    print("att_env_train shape:", env1_train.shape)
    print("unatt_env_train shape:", env2_train.shape)
    print("eeg_valid shape:", eeg_valid.shape)
    print("att_env_valid shape:", env1_valid.shape)
    print("unatt_env_valid shape:", env2_valid.shape)

    print(labels_train.shape)
