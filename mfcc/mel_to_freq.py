import numpy as np

def mel_to_freq(m):
    '''
    Get the frequency from the mel scale

    Parameters
    ----------
    m : int
        Mel value

    Returns
    -------
    f : int
        Frequency    
    '''
    return 700 * (np.exp(m/1125) - 1)