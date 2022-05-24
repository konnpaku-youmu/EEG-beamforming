import tensorflow as tf
from tensorflow import keras
import scipy.io as sio
import numpy as np

sub_num=1
trial_num=1

eeg_data = sio.loadmat("eeg_data/eeg_data/1/6/eeg.mat")
eeg_data = eeg_data['eeg']

env1_data = sio.loadmat("eeg_data/eeg_data/1/6/env1.mat")
env1_data = env1_data['env1']

env2_data = sio.loadmat("eeg_data/eeg_data/1/6/env2.mat")
env2_data = env2_data['env2']

mapping = sio.loadmat("eeg_data/eeg_data/1/6/mapping.mat")
mapping = mapping['mapping'][0]

locus_data = sio.loadmat("eeg_data/eeg_data/1/3/Locus.mat")
locus_data = list(map(lambda x: mapping.index(x), locus_data['locus']))

for seg in range(0,5):
    model = keras.models.load_model('./models/fine-tuned models_LSTM_5sec_64hz/LSTM_sub'+str(sub_num)+'.h5')
    eeg = eeg_data[:, :, seg]
    env1= env1_data[:, :, seg]
    env2= env2_data[:, :, seg]
    locus_seg = locus_data[seg*320:seg*320+320-1]
    locus_label = np.round(np.mean(locus_seg))
    eeg=np.expand_dims(eeg,0)
    env1=np.expand_dims(env1,0)
    env2=np.expand_dims(env2,0)

    ynew = int(model.predict([eeg, env1 ,env2])[0, 0] > 0.5)

    attention = mapping[ynew]

    print(attention)
    


