import numpy as np
import librosa
from scipy import signal, fft

from . import freq_bins, filterbank, deltas

def get_mfcc(x: np.ndarray,
             fx: int,
             window: str = 'hann',
             window_len: int = 0.025,
             window_step: int = 0.01,
             n_filters: int = 26,
             n_cepstral: int = 12,
             mfft: int = 512,
             calculate_deltas: bool = True,
             calculate_ddeltas: bool = True) -> np.ndarray:
    '''
    Get the MFCC of a signal

    Parameters
    ----------
    path : str
        Path to the signal
    window : str
        Window to be applied to the signal
    window_len : int
        Window length in seconds
    window_step : int
        Window step in seconds
    n_filters : int
        Number of filters
    n_cepstral : int
        Number of cepstral coefficients
    mfft : int
        Number of points in the FFT
    calculate_deltas : bool
        If True, calculate the deltas
    calculate_ddeltas : bool
        If True, calculate the delta-deltas
    
    Returns
    -------
    mfcc : numpy array
        MFCC of the signal
    '''
    # Check if the signal length is enough
    if len(x) < 512:
        x = librosa.util.fix_length(x, size=512) # Pad with 0s
    
    # Apply STFT
    w = signal.get_window(window=window, Nx=int(window_len*fx))
    STFT = signal.ShortTimeFFT(win=w, hop=int(window_step*fx), fs=fx, scale_to='magnitude',mfft=mfft)  
    Sx = STFT.stft(x)

    # Calculate Periodogram estimate of the power spectrum
    Sx_power = (np.abs(Sx)**2) / int(window_len*fx)

    # Get the frequency bins
    f_bins = freq_bins(lower_freq=0, upper_freq=max(8000,int(fx/2)), n_filters=n_filters, fx=fx, mfft=mfft)

    # Filterbank
    filterbank_ = filterbank(f_bins=f_bins, n_filters=n_filters, mfft=mfft)

    # Calculate the filterbank energy
    filterbank_energies = np.dot(filterbank_, Sx_power)

    # Take the logarithm
    log_filterbank_energies = np.log(filterbank_energies)

    # Take the DCT
    mfcc = fft.dct(log_filterbank_energies, type=2, axis=0, norm='ortho')[:n_cepstral]

    # Calculate the deltas
    if calculate_deltas:
        delta = deltas(mfcc, window=2)
        mfcc = np.vstack([mfcc, delta])

    # Calculate the delta-deltas
    if calculate_ddeltas:
        delta_delta = deltas(delta, window=2)
        mfcc = np.vstack([mfcc, delta_delta])

    return mfcc