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

time_window=1400
layers=3
kernel_size=3
filters_spatial=8
filters_dilated=16

eeg  = tf.keras.layers.Input(shape=[time_window, 64])
env1 = tf.keras.layers.Input(shape=[time_window, 1])
env2 = tf.keras.layers.Input(shape=[time_window, 1])

#add model layers
## ---- add your code ----here

# Activations to apply
activations = ["relu"] * layers

# Spatial convolution
env1_conv = env1
env2_conv = env2
eeg_conv = tf.keras.layers.Conv1D(filters_spatial, kernel_size=1)(eeg)

# Dilated convolution
for l in range(layers):
  # eeg
  eeg_conv  = tf.keras.layers.Conv1D(filters_dilated, kernel_size=kernel_size, dilation_rate=kernel_size ** l,
                                            strides=1, activation=activations[l])(eeg_conv)
                                            
  # env 
  env_conv = tf.keras.layers.Conv1D(filters_dilated, kernel_size=kernel_size, dilation_rate=kernel_size ** l,
                                                strides=1, activation=activations[l])
  env1_conv = env_conv(env1_conv)
  env2_conv = env_conv(env2_conv)


# Classification
cos1 = tf.keras.layers.Dot(1, normalize=True)([eeg_conv, env1_conv])
cos2 = tf.keras.layers.Dot(1, normalize=True)([eeg_conv, env2_conv])
cos_similarity = tf.keras.layers.Concatenate()([cos1, cos2])
cos_flat = tf.keras.layers.Flatten()(cos_similarity)
out1 = tf.keras.layers.Dense(1, activation="sigmoid")(cos_flat)

# 1 output per batch
#out = tf.keras.layers.Reshape([1], name=output_name)(out1)
model = tf.keras.Model(inputs=[eeg, env1, env2], outputs=[out1])

model.summary()

def batch_equalizer(eeg, env1, env2, labels):
    # present each of the eeg segments twice, where the envelopes, and thus the labels 
    # are swapped around. EEG presented in small segments [bs, window_length, 64]
    return np.concatenate([eeg,eeg], axis=0), np.concatenate([env1, env2], axis=0),np.concatenate([env2, env1], axis=0), np.concatenate([labels, (labels+1)%2], axis=0)

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
eeg_train,env1_train,env2_train, labels_train= batch_equalizer(eeg_train, env1_train, env2_train, labels_train)
eeg_valid,env1_valid,env2_valid, labels_valid= batch_equalizer(eeg_valid, env1_valid, env2_valid, labels_valid)

print(eeg_train.shape)
print(env1_train.shape)
print(env2_train.shape)
print(labels_train.shape)

print(eeg_valid.shape)
print(env1_valid.shape)
print(env2_valid.shape)
print(labels_valid.shape)

earlystop = EarlyStopping(monitor='val_loss',
                              patience=10,
                              verbose=0, mode='min')

model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      metrics=["acc"],
      loss=["binary_crossentropy"])

history = model.fit([eeg_train, env1_train, env2_train], labels_train,batch_size=64,
          epochs=100,validation_data=([eeg_valid, env1_valid, env2_valid], labels_valid),
          shuffle=True,
          verbose=2,callbacks=[earlystop])

print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss')
plt.ylabel('loss')  
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('accuracy')
plt.ylabel('acc')  
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()