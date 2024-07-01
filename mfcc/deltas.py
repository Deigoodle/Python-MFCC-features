import numpy as np

def deltas(mfcc, window=2):
    '''
    Compute the delta coefficients of the MFCC

    Parameters
    ----------
    mfcc : numpy array
        MFCC coefficients
    window : int
        Window size
        
    Returns
    -------
    delta : numpy array
        Delta coefficients
    '''
    T = mfcc.shape[1]
    delta = np.zeros_like(mfcc)
    for t in range(T):
        numerator = 0
        denominator = 0
        for w in range(1, window+1):
            numerator += w * (mfcc[:, min(T-1, t+w)] - mfcc[:, max(0, t-w)])
            denominator += w**2
        delta[:, t] = numerator / (2 * denominator)
    return delta