# -*- coding: utf-8 -*-
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

def batch_equalizer(eeg, env1, env2, labels):
    # present each of the eeg segments twice, where the envelopes, and thus the labels 
    # are swapped around. EEG presented in small segments [bs, window_length, 64]
    return np.concatenate([eeg,eeg], axis=0), np.concatenate([env1, env2], axis=0),np.concatenate([env2, env1], axis=0), np.concatenate([labels, (labels+1)%2], axis=0)

def load_large_mat(filepath):
    arrays = {}
    f = h5py.File("./data_640_40-10/"+filepath)
    for k, v in f.items():
        arrays[k] = np.array(v)
    f.close()
    return arrays

eeg_train    = load_large_mat('eeg_640_train.mat')['eeg_640_train']
env1_train   = load_large_mat('env1_640_train.mat')['env1_640_train']
env2_train   = load_large_mat('env2_640_train.mat')['env2_640_train']

eeg_valid    = load_large_mat('eeg_640_valid.mat')['eeg_640_valid']
env1_valid   = load_large_mat('env1_640_valid.mat')['env1_640_valid']
env2_valid   = load_large_mat('env2_640_valid.mat')['env2_640_valid']

eeg_train  = eeg_train.transpose((0,2,1))  
env1_train = env1_train.transpose((0,2,1))  
env2_train = env2_train.transpose((0,2,1))  

eeg_valid  = eeg_valid.transpose((0,2,1))  
env1_valid = env1_valid.transpose((0,2,1))  
env2_valid = env2_valid.transpose((0,2,1))

'''
eeg_train  = np.reshape(eeg_train  , (eeg_train.shape[0]  ,eeg_train.shape[2]   ,eeg_train.shape[1]))
env1_train = np.reshape(env1_train , (env1_train.shape[0] ,env1_train.shape[2]  ,env1_train.shape[1]))
env2_train = np.reshape(env2_train , (env2_train.shape[0] ,env2_train.shape[2]  ,env2_train.shape[1]))

eeg_valid  = np.reshape(eeg_valid  , (eeg_valid.shape[0]  ,eeg_valid.shape[2]   ,eeg_valid.shape[1]))
env1_valid = np.reshape(env1_valid , (env1_valid.shape[0] ,env1_valid.shape[2]  ,env1_valid.shape[1]))
env2_valid = np.reshape(env2_valid , (env2_valid.shape[0] ,env2_valid.shape[2]  ,env2_valid.shape[1]))
'''

print(eeg_train.shape)
print(env1_train.shape)
print(env2_train.shape)
print(eeg_valid.shape)
print(env1_valid.shape)
print(env2_valid.shape)

eeg_train[0,:,0]

eeg_mean  = np.expand_dims(np.mean(np.mean(np.concatenate((eeg_train,eeg_valid),axis=0),axis=0),axis=0),axis=[0,1])
env1_mean = np.expand_dims(np.mean(np.mean(np.concatenate((env1_train,env1_valid),axis=0),axis=0),axis=0),axis=[0,1])
env2_mean = np.expand_dims(np.mean(np.mean(np.concatenate((env2_train,env2_valid),axis=0),axis=0),axis=0),axis=[0,1])

eeg_max  = np.expand_dims(np.max(np.max(np.concatenate((eeg_train,eeg_valid),axis=0),axis=0),axis=0),axis=[0,1])
env1_max = np.expand_dims(np.max(np.max(np.concatenate((env1_train,env1_valid),axis=0),axis=0),axis=0),axis=[0,1])
env2_max = np.expand_dims(np.max(np.max(np.concatenate((env2_train,env2_valid),axis=0),axis=0),axis=0),axis=[0,1])

eeg_min  = np.expand_dims(np.min(np.min(np.concatenate((eeg_train,eeg_valid),axis=0),axis=0),axis=0),axis=[0,1])
env1_min = np.expand_dims(np.min(np.min(np.concatenate((env1_train,env1_valid),axis=0),axis=0),axis=0),axis=[0,1])
env2_min = np.expand_dims(np.min(np.min(np.concatenate((env2_train,env2_valid),axis=0),axis=0),axis=0),axis=[0,1])

eeg_train  = (eeg_train  - eeg_mean)   / (eeg_max-eeg_min)
env1_train = (env1_train - env1_mean)  / (env1_max-env1_min)
env2_train = (env2_train  - env2_mean) / (env2_max-env2_min)

eeg_valid  = (eeg_valid  - eeg_mean)  / (eeg_max-eeg_min)
env1_valid = (env1_valid - env1_mean) / (env1_max-env1_min)
env2_valid = (env2_valid - env2_mean) / (env2_max-env2_min)

'''
eeg_mean = np.expand_dims(np.mean(eeg_train,axis=1),axis=1)
eeg_var  = np.expand_dims(np.var(eeg_train,axis=1),axis=1)
eeg_train=(eeg_train - eeg_mean)/ eeg_var;

env1_mean = np.expand_dims(np.mean(env1_train,axis=1),axis=1)
env1_var  = np.expand_dims(np.var(env1_train,axis=1),axis=1)
env1_train=(env1_train - env1_mean)/ env1_var;

env2_mean = np.expand_dims(np.mean(env2_train,axis=1),axis=1)
env2_var  = np.expand_dims(np.var(env2_train,axis=1),axis=1)
env2_train=(env2_train - env2_mean)/ env2_var;


eeg_mean = np.expand_dims(np.mean(eeg_valid,axis=1),axis=1)
eeg_var  = np.expand_dims(np.var(eeg_valid,axis=1),axis=1)
eeg_valid=(eeg_valid - eeg_mean)/ eeg_var;

env1_mean = np.expand_dims(np.mean(env1_valid,axis=1),axis=1)
env1_var  = np.expand_dims(np.var(env1_valid,axis=1),axis=1)
env1_valid=(env1_valid - env1_mean)/ env1_var;

env2_mean = np.expand_dims(np.mean(env2_valid,axis=1),axis=1)
env2_var  = np.expand_dims(np.var(env2_valid,axis=1),axis=1)
env2_valid=(env2_valid - env2_mean)/ env2_var;
'''

labels_valid = np.zeros((eeg_valid.shape[0],1)).astype(int)
labels_train = np.zeros((eeg_train.shape[0],1)).astype(int)
eeg_train,env1_train,env2_train, labels_train= batch_equalizer(eeg_train, env1_train, env2_train, labels_train)
eeg_valid,env1_valid,env2_valid, labels_valid= batch_equalizer(eeg_valid, env1_valid, env2_valid, labels_valid)

# reshape to fit the model
eeg_train = np.reshape(eeg_train , (eeg_train.shape[0] , 8, 8, eeg_train.shape[1]  ,eeg_train.shape[2]//64))
eeg_valid = np.reshape(eeg_valid , (eeg_valid.shape[0] , 8, 8, eeg_valid.shape[1]  ,eeg_valid.shape[2]//64))

print(eeg_train.shape)
print(env1_train.shape)
print(env2_train.shape)
print(labels_train.shape)
print(eeg_valid.shape)
print(env1_valid.shape)
print(env2_valid.shape)
print(labels_valid.shape)

time_window = 640;
channel_num=64;

eeg  = tf.keras.layers.Input(shape=[8, 8, time_window, channel_num // 64])
env1 = tf.keras.layers.Input(shape=[time_window, 1])
env2 = tf.keras.layers.Input(shape=[time_window, 1])

#add model layers
filters = 1
kernel_size = 8
eeg_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same', dilation_rate=(10, 10))(eeg)
eeg_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same', dilation_rate=(2, 2))(eeg_conv)
# reshape to 640*64
eeg_conv = tf.keras.layers.Reshape((640,64))(eeg_conv)

eeg_conv = tf.keras.layers.Conv1D(filters, kernel_size=kernel_size,padding='same')(eeg_conv)

cos1 = tf.keras.layers.Dot(1,normalize= True)([eeg_conv , env1])
cos2 = tf.keras.layers.Dot(1,normalize= True)([eeg_conv , env2])
#cos1 = tf.keras.layers.Dot(1,normalize= True)([eeg_conv , env1[:,:-(kernel_size-1),:]])
#cos2 = tf.keras.layers.Dot(1,normalize= True)([eeg_conv , env2[:,:-(kernel_size-1),:]])

# Classification
cos_similarity = tf.keras.layers.Concatenate()([cos1, cos2])
cos_flat = tf.keras.layers.Flatten()(cos_similarity)
out1 = tf.keras.layers.Dense(1, activation="sigmoid")(cos_flat)

# 1 output per batch
model = tf.keras.Model(inputs=[eeg, env1, env2], outputs=[out1])

model.summary()

earlystop = EarlyStopping(monitor='val_loss',
                              patience=20,
                              verbose=0, mode='min')

model.compile(
      optimizer=tf.keras.optimizers.Adam(lr=0.01),
      metrics=["acc"],
      loss=["binary_crossentropy"])

history = model.fit([eeg_train, env1_train, env2_train], labels_train,batch_size=64,
          epochs=200,validation_data=([eeg_valid, env1_valid, env2_valid], labels_valid),
          shuffle=True,
          verbose=2,callbacks=[earlystop])

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

# """# subject dependent """

# # again load, reshape, normalize dataset

# sub_eeg_train=np.zeros((10,48*4,640,64))
# sub_env1_train=np.zeros((10,48*4,640,1))
# sub_env2_train=np.zeros((10,48*4,640,1))
# sub_eeg_valid=np.zeros((10,48*1,640,64))
# sub_env1_valid=np.zeros((10,48*1,640,1))
# sub_env2_valid=np.zeros((10,48*1,640,1))

# print(sub_eeg_train.shape)
# print(sub_env1_train.shape)
# print(sub_env2_train.shape)
# print(sub_eeg_valid.shape)
# print(sub_env1_valid.shape)
# print(sub_env2_valid.shape)


# for i in range(0,10):
#   sub_eeg_train[i,:,:,:] = eeg_train[i*192:i*192+192,:,:]
#   sub_env1_train[i,:,:,:] = env1_train[i*192:i*192+192,:,:]
#   sub_env2_train[i,:,:,:] = env2_train[i*192:i*192+192,:,:]

#   sub_eeg_valid[i,:,:,:] = eeg_valid[i*48:i*48+48,:,:]
#   sub_env1_valid[i,:,:,:] = env1_valid[i*48:i*48+48,:,:]
#   sub_env2_valid[i,:,:,:] = env2_valid[i*48:i*48+48,:,:]




# for i in range(0,9):
#   eeg_train  = sub_eeg_train[i,:,:,:]
#   env1_train = sub_env1_train[i,:,:,:]
#   env2_train = sub_env2_train[i,:,:,:]
#   eeg_valid  = sub_eeg_valid[i,:,:,:]
#   env1_valid = sub_env1_valid[i,:,:,:]
#   env2_valid = sub_env2_valid[i,:,:,:]
#   labels_valid = np.zeros((eeg_valid.shape[0],1)).astype(int)
#   labels_train = np.zeros((eeg_train.shape[0],1)).astype(int)
#   eeg_train,env1_train,env2_train, labels_train= batch_equalizer(eeg_train, env1_train, env2_train, labels_train)
#   eeg_valid,env1_valid,env2_valid, labels_valid= batch_equalizer(eeg_valid, env1_valid, env2_valid, labels_valid)


#   earlystop = EarlyStopping(monitor='val_loss',
#                               patience=10,
#                               verbose=0, mode='min')

#   model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#         metrics=["acc"],
#         loss=["binary_crossentropy"])

#   history = model.fit([eeg_train, env1_train, env2_train], labels_train,batch_size=64,
#             epochs=50,validation_data=([eeg_valid, env1_valid, env2_valid], labels_valid),
#             shuffle=True,
#             verbose=2,callbacks=[earlystop])
  
#   print(history.history.keys())

#   #plt.subplot(1,2,1)        
#   plt.plot(history.history['loss'])
#   plt.plot(history.history['val_loss'])
#   plt.title('loss')
#   plt.ylabel('loss')  
#   plt.xlabel('epoch')
#   plt.legend(['train', 'valid'], loc='upper left')
#   plt.show()

#   plt.plot(history.history['acc'])
#   plt.plot(history.history['val_acc'])
#   plt.title('accuracy')
#   plt.ylabel('acc')  
#   plt.xlabel('epoch')
#   plt.legend(['train', 'valid'], loc='upper left')
#   plt.show()