import numpy as np

def freq_to_mel(f):
    '''
    Get the mel scale from the frequency

    Parameters
    ----------
    f : int
        Frequency

    Returns
    -------
    m : int
        Mel value    
    '''
    return 1125 * np.log(1 + f/700)