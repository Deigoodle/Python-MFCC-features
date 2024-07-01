import numpy as np

def filterbank(f_bins, n_filters, mfft):
    '''
    Create a filterbank to be applied to the STFT

    Parameters
    ----------
    f_bins : numpy array
        Array with the frequency bins
    n_filters : int
        Number of filters
    mfft : int
        Number of points in the FFT

    Returns
    -------
    filterbank : numpy array
        Filterbank to be applied to the STFT
    '''
    filterbank = np.zeros((n_filters, mfft//2+1)) # if mfft = 512 -> (n_filters, 257) 

    for i in range(len(f_bins)-2):

        for j in range(f_bins[i], f_bins[i+1]):
            filterbank[i, j] = (j - f_bins[i]) / (f_bins[i+1] - f_bins[i])

        for j in range(f_bins[i+1], f_bins[i+2]):
            filterbank[i, j] = (f_bins[i+2] - j) / (f_bins[i+2] - f_bins[i+1])
            
    return filterbank