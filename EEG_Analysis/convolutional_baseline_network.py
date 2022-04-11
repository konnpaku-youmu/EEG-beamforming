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

time_window = 640;
eeg = tf.keras.layers.Input(shape=[time_window, 64])
env1 = tf.keras.layers.Input(shape=[time_window, 1])
env2 = tf.keras.layers.Input(shape=[time_window, 1])

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#add model layers
## ---- add your code ----here
filters = 1
kernel_size = 16
eeg_conv = tf.keras.layers.Conv1D(filters, kernel_size=kernel_size)(eeg)
cos1 = tf.keras.layers.Dot(1,normalize= True)([eeg_conv , env1[:,:-(kernel_size-1),:]])
cos2 = tf.keras.layers.Dot(1,normalize= True)([eeg_conv , env2[:,:-(kernel_size-1),:]])

# Classification
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

eeg_path = '/content/drive/MyDrive/Colab Notebooks/P&D_phase2/data/modified data/eeg_data.mat'
env_attend_path = '/content/drive/MyDrive/Colab Notebooks/P&D_phase2/data/modified data/env1_data.mat'
env_unattend_path = '/content/drive/MyDrive/Colab Notebooks/P&D_phase2/data/modified data/env2_data.mat'

from scipy.io import loadmat

eeg  = loadmat(eeg_path);
env1 = loadmat(env_attend_path)
env2 = loadmat(env_unattend_path)
eeg = eeg['eeg_data']
env1 = env1['env1_data']
env2 = env2['env2_data']

print(eeg.shape)
print(env1.shape)
print(env2.shape)

import random
valid_index = random.sample(range(0,eeg.shape[0]),int(0.2*eeg.shape[0]))
x = np.array(range(0,eeg.shape[0]))
train_index = np.delete(x,valid_index)

np.size(train_index)

np.size(valid_index)

eeg_valid  = [eeg[index][:][:]   for index in valid_index]
env1_valid = [env1[index][:][:]  for index in valid_index]
env2_valid = [env2[index][:][:]  for index in valid_index]

eeg_valid = np.array(eeg_valid)
env1_valid = np.array(env1_valid)
env2_valid = np.array(env2_valid)
print(eeg_valid.shape)
print(env1_valid.shape)
print(env2_valid.shape)

eeg_train  = [eeg[index][:][:]   for index in train_index]
env1_train = [env1[index][:][:]  for index in train_index]
env2_train = [env2[index][:][:]  for index in train_index]

eeg_train = np.array(eeg_train)
env1_train = np.array(env1_train)
env2_train = np.array(env2_train)
print(eeg_train.shape)
print(env1_train.shape)
print(env2_train.shape)

from scipy.io import savemat

savemat('eeg_train.mat',  {'eeg_train': eeg_train})
savemat('env1_train.mat', {'env1_train':env1_train})
savemat('env2_train.mat', {'env2_train': env2_train})


savemat('eeg_valid.mat',  {'eeg_valid': eeg_valid})
savemat('env1_valid.mat', {'env1_valid': env1_valid})
savemat('env2_valid.mat', {'env2_valid': env2_valid})

"""### **Load modified data**"""

from scipy.io import loadmat

eeg_train_path  = './data/modified data 640_train_valid/eeg_train.mat'
env1_train_path = './data/modified data 640_train_valid/env1_train.mat'
env2_train_path = './data/modified data 640_train_valid/env2_train.mat'
eeg_valid_path  = './data/modified data 640_train_valid/eeg_valid.mat'
env1_valid_path = './data/modified data 640_train_valid/env1_valid.mat'
env2_valid_path = './data/modified data 640_train_valid/env2_valid.mat'

eeg_train    = loadmat(eeg_train_path);
env1_train   = loadmat(env1_train_path)
env2_train   = loadmat(env2_train_path)

eeg_train    = eeg_train['eeg_train']
env1_train   = env1_train['env1_train']
env2_train   = env2_train['env2_train']

eeg_valid    = loadmat(eeg_valid_path);
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

model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      metrics=["acc"],
      loss=["binary_crossentropy"])

history = model.fit([eeg_train, env1_train, env2_train], labels_train,batch_size=64,
          epochs=100,validation_data=([eeg_valid, env1_valid, env2_valid], labels_valid),
          shuffle=True,
          verbose=2)

print(history.history.keys())

#plt.subplot(1,2,1)        
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