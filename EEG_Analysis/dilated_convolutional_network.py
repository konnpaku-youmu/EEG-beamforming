# -*- coding: utf-8 -*-
# Load required libraries
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Activation
from keras import regularizers
import tensorflow as tf 
from tensorflow.python.ops.linalg_ops import norm
import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

from scipy.io import loadmat

eeg_train_path  = './data/modified data 1400_train_valid/eeg_train.mat'
env1_train_path = './data/modified data 1400_train_valid/env1_train.mat'
env2_train_path = './data/modified data 1400_train_valid/env2_train.mat'

eeg_valid_path  = './data/modified data 1400_train_valid/eeg_valid.mat'
env1_valid_path = './data/modified data 1400_train_valid/env1_valid.mat'
env2_valid_path = './data/modified data 1400_train_valid/env2_valid.mat'

eeg_train    = loadmat(eeg_train_path)
env1_train   = loadmat(env1_train_path)
env2_train   = loadmat(env2_train_path)

eeg_train    = eeg_train['eeg_train']
env1_train   = env1_train['env1_train']
env2_train   = env2_train['env2_train']

eeg_valid    = loadmat(eeg_valid_path)
env1_valid   = loadmat(env1_valid_path)
env2_valid   = loadmat(env2_valid_path)

eeg_valid    = eeg_valid['eeg_valid']
env1_valid   = env1_valid['env1_valid']
env2_valid   = env2_valid['env2_valid']

labels_valid = np.zeros((eeg_valid.shape[0],1)).astype(int)
labels_train = np.zeros((eeg_train.shape[0],1)).astype(int)

