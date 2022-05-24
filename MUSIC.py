from math import ceil
import matplotlib.pyplot as plt
import scipy
import numpy as np
from scipy.signal import stft

def cfar(x, num_train, num_guard, rate_fa):
    """
    Detect peaks with CFAR algorithm.
    
    num_train: Number of training cells.
    num_guard: Number of guard cells.
    rate_fa: False alarm rate. 
    """
    num_cells = x.size
    num_train_half = round(num_train / 2)
    num_guard_half = round(num_guard / 2)
    num_side = num_train_half + num_guard_half
 
    alpha = num_train*(rate_fa**(-1/num_train) - 1) # threshold factor
    
    peak_idx = []
    peak_mag = []
    for i in range(num_side, num_cells - num_side):
        
        if i != i-num_side+np.argmax(x[i-num_side:i+num_side+1]): 
            continue
        
        sum1 = np.sum(x[i-num_side:i+num_side+1])
        sum2 = np.sum(x[i-num_guard_half:i+num_guard_half+1])
        p_noise = (sum1 - sum2) / num_train
        threshold = alpha * p_noise
        
        if x[i] > threshold: 
            peak_idx.append(i)
            peak_mag.append(x[i])
    
    # sort index by magnitude
    peak_idx = np.array(peak_idx)
    peak_mag = np.abs(np.array(peak_mag))
    peak_idx = peak_idx[peak_mag.argsort()[::-1]]
  
    return peak_idx


def doa_esti_MUSIC(mic, rir_info, doa_method="wideband", stft_length=1024, stft_overlap=200):

    # assert mic is not None
    try:
        assert mic is not None
    except AssertionError:
        print('No microphone data provided. Aborting...')
        return

    # assert doa_method in ["wideband", "narrowband"]
    try:
        assert doa_method.lower() in ["wideband", "narrowband"]
    except AssertionError:
        print('Invalid doa method. Aborting...')
        return

    # assert RIR sampling freq is 44100
    try:
        assert rir_info.sample_rate == 16000 or rir_info.sample_rate == 44100
    except AssertionError:
        print('Invalid RIR sampling freq. Aborting...')
        return

    win = scipy.signal.windows.hamming(300)
    freq, time, Zxx = stft(mic, fs=rir_info.sample_rate, window=win,
                            nfft=stft_length, nperseg=300, noverlap=stft_overlap, 
                            axis=0)
    
    # permute mic axis to last
    spectrogram = np.transpose(Zxx, (0, 2, 1))
    
    v_sound = 340.0
    # inter-microphone distance
    d =  rir_info.get_intermic_dist()
    # time diff between each mic
    theta = np.linspace(np.linspace(0, 180, 361), np.linspace(0, 180, 361), d.shape[1]).dot(np.pi / 180)
    tau = d.T * np.cos(theta) / v_sound

    bin_range = np.arange(0, spectrogram.shape[0])
    freq_range = 2 * bin_range / stft_length * (rir_info.sample_rate / 2)
    omega_range = 2 * np.pi * freq_range

    cutoff_freq = ceil(stft_length // 2)
    p_mean = np.ones(theta.shape[1], dtype=np.float64)

    for freq_bin in range(1, cutoff_freq):

        # channel covariance
        R = np.matmul(spectrogram[freq_bin, :, :].transpose().conj(), spectrogram[freq_bin, :, :])

        # Eigenvalue decomposition
        eig_val_R, eig_vec_R = np.linalg.eig(R)
        # Eigenvalues sorted in ascending order
        idx = eig_val_R.argsort()[::1]
        eig_val_R = eig_val_R[idx]
        eig_vec_R = eig_vec_R[:, idx]

        null_dims = rir_info.num_mics - rir_info.num_sources
        # select the first null_dims eigenvectors
        E = eig_vec_R[:, :null_dims]

        # define the steering matrix
        G = np.exp(-1j * omega_range[freq_bin] * tau)
        p = 1 / np.diag(np.matmul(G.conj().T, np.matmul(E, np.matmul(E.conj().T, G))))
        
        p_mean = p_mean + p
        # p_mean = p_mean * np.power(np.abs(p), 1 / ((cutoff_freq - 1) ** 2))

    p_mean = p_mean / (cutoff_freq - 1)

    # make false alarm rate changing with reverb time
    t60 = rir_info.get_reverb_time()
    rate_fa_cond = None

    if t60 < 0.2:
        rate_fa_cond = 0.3
    elif t60 < 0.5:
        rate_fa_cond = 0.5
    elif t60 < 1:
        rate_fa_cond = 0.7
    else:
        rate_fa_cond = 0.9

    # CFAR detection
    # peak_idx = cfar(p_mean, num_train=60, num_guard=30, rate_fa=rate_fa_cond)
    peak_idx = cfar(p_mean, num_train=5, num_guard=10, rate_fa=0.8)

    # peaks, _ = scipy.signal.find_peaks(p_mean, distance=20)

    return theta[0, :] * (180 / np.pi), p_mean, peak_idx[:rir_info.num_sources]
