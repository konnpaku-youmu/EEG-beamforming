from typing import Mapping
from util_tools import *
from FD_GSC import *

import keras


def read_eegdata(sub_num, trial_num):
    eeg_data = sio.loadmat(
        "eeg_data/eeg_data/{0}/{1}/eeg.mat".format(sub_num, trial_num))
    eeg_data = eeg_data['eeg']

    env1_data = sio.loadmat(
        "eeg_data/eeg_data/{0}/{1}/env1.mat".format(sub_num, trial_num))
    env1_data = env1_data['env1']

    env2_data = sio.loadmat(
        "eeg_data/eeg_data/{0}/{1}/env2.mat".format(sub_num, trial_num))
    env2_data = env2_data['env2']

    mapping = sio.loadmat(
        "eeg_data/eeg_data/{0}/{1}/mapping.mat".format(sub_num, trial_num))
    mapping = mapping['mapping'][0]

    locus_data = sio.loadmat(
        "eeg_data/eeg_data/{0}/{1}/Locus.mat".format(sub_num, trial_num))
    locus_data = list(map(lambda x: mapping.index(x), locus_data['locus']))

    return eeg_data, env1_data, env2_data, locus_data, mapping


if __name__ == "__main__":
    RIRs = RIR_Info("Data/RIRs_16kHz/LMA_rirs.mat")

    sub_num = 16
    trial_num = 6

    eeg_data, env1_data, env2_data, locus_data, mapping = read_eegdata(
        sub_num, trial_num)
    fs, audio = load_audios(sub_num, trial_num)

    eeg_wsize = eeg_data.shape[0]
    audio_wsize = audio.shape[0] // 6

    speech = []

    for seg in range(0, 6):
        model = keras.models.load_model(
            "./models/fine-tuned models_LSTM_5sec_64hz/LSTM_sub{}.h5".format(sub_num))

        eeg = eeg_data[:, :, seg]
        env1 = env1_data[:, :, seg]
        env2 = env2_data[:, :, seg]
        locus_seg = locus_data[seg*eeg_wsize:(seg+1)*eeg_wsize]
        locus_label = np.round(np.mean(locus_seg))
        eeg = np.expand_dims(eeg, 0)
        env1 = np.expand_dims(env1, 0)
        env2 = np.expand_dims(env2, 0)

        ynew = int(model.predict([eeg, env1, env2])[0, 0] > 0.5)

        attention = mapping[ynew]

        print("Attention: {0}\t Hit={1}".format(
            attention, ynew == locus_label))

        # DOA
        audio_seg = audio[seg*audio_wsize:(seg+1)*audio_wsize, :]
        theta, pseudo_spectrum, peaks = doa_esti_MUSIC(audio_seg, RIRs)

        DOA_esti = theta[peaks] - 90
        DOA_attention = None

        if 'F' not in mapping:
            if attention == 'L':
                DOA_attention = min(DOA_esti)
            elif attention == 'R':
                DOA_attention = max(DOA_esti)
        elif 'L' not in mapping:
            if attention == 'F':
                DOA_attention = min(DOA_esti)
            elif attention == 'R':
                DOA_attention = max(DOA_esti)
        elif 'R' not in mapping:
            if attention == 'L':
                DOA_attention = min(DOA_esti)
            elif attention == 'F':
                DOA_attention = max(DOA_esti)

        print("DOA: {0}".format(DOA_esti))
        print("DOA_attention: {0}".format(DOA_attention))
        rir_idx = np.argmin(abs(RIRs.RIR_angles - DOA_attention))

        _, speeches = fd_gsc(audio_seg, RIRs, rir_idx=rir_idx)

        print("Speech_sep: {0}".format(speeches[0].shape))

        # speeches[0] = speeches[0] / np.max(np.abs(speeches[0]))

        speech.append(speeches[0])
    
    speech = np.concatenate(speech, axis=0)

    wavfile.write("./audio_outputs/speech_est.wav", RIRs.sample_rate, speech)