from util_tools import *
from FD_GSC import *

doa_lut = load_timevar_angles()



fs, RIRs = load_RIRs()
fs_audio, speech, ref_l, ref_r = load_audios()
rir_lut = [[pos[0], np.stack((RIRs[pos[1]],RIRs[pos[2]]), axis=2)] for pos in doa_lut]

try:
    assert fs == fs_audio
except AssertionError:
    print('Sample rate of RIRs and audio files do not match. Aborting...')

rir_info = RIR_Info()
rir_info.mini_load(fs, None, speech.shape[1])

gsc_len = 2
GSC = FD_GSC(rir_info, win_len = gsc_len*fs, fft_len=1024)

pos_switch_cnt = 0
GSC.update_rir(rir_lut[pos_switch_cnt][1])

for k, sample in enumerate(speech):
    time_elapsed = k / fs

    if time_elapsed >= doa_lut[pos_switch_cnt][0]:
        pos_switch_cnt += 1
        GSC.update_rir(rir_lut[pos_switch_cnt][1])

    GSC.update_sample(sample)

    if k % fs == 0 and k >= gsc_len*fs:
        print(k/fs)
        GSC.step()
        plt.subplot(2,1,1)
        GSC.plot_all()
        plt.subplot(2,1,2)
        plt.plot(ref_l[k-gsc_len*fs:k, 0])
        plt.plot(ref_r[k-gsc_len*fs:k, 0])
        plt.show()
