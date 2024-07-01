import numpy as np
from .freq_to_mel import freq_to_mel
from .mel_to_freq import mel_to_freq

def freq_bins(lower_freq, upper_freq, n_filters, fx, mfft):
    '''
    Get the frequency bins for the filterbank

    Parameters
    ----------
    lower_freq : int
        Lower frequency of the filterbank
    upper_freq : int
        Upper frequency of the filterbank
    n_filters : int
        Number of filters
    fx : int
        Sampling frequency
    mfft : int
        Number of points in the FFT

    Returns
    -------
    f_bins : numpy array
        Array with the frequency bins
    '''
    # Get the mel scale of the lower and upper frequencies
    lower_mel = freq_to_mel(lower_freq)
    upper_mel = freq_to_mel(upper_freq)

    # Get the mel bins
    mel_bins = np.linspace(lower_mel, upper_mel, n_filters+2)

    # Convert the mel bins to frequency bins
    freq_bins = mel_to_freq(mel_bins)

    # Convert the frequency bins to the bins in the FFT
    f_bins = np.floor(freq_bins * mfft / fx).astype(int)
    
    return f_bins