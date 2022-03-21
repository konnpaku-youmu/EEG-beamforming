import glob
import numpy as np
import scipy

def load_eeg_filenames(path):
    """
    Loads EEG filenames from a path.
    """
    filenames = glob.glob(path+"/*.mat")
    print('Found {0} mat files in {1}'.format(len(filenames), path))
    return filenames

def load_eeg_mat(subject_id):
    """
    Loads EEG data from a mat file.
    """
    eeg_filenames = load_eeg_filenames("./dataset/EEG/{:0>3d}".format(subject_id))
    eeg_data = np.ndarray((0,0))
    for matfile in eeg_filenames:
        mat_struct = scipy.io.loadmat(matfile)
        eeg_data = mat_struct['trial']['RawData'][0,0]['EegData'][0,0]
    return eeg_data

if __name__ == '__main__':
    # Load EEG data
    for subject_id in range(1, 38):
        load_eeg_mat(subject_id)
