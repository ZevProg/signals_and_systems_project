## decimation and interpolation
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def down_sample(x, M):
    return x[::M]

def decimation(x, M):
    h = signal.firwin(1024, cutoff=1 / M, window='hamming')
    y = signal.convolve(x, h, mode='same')
    y = down_sample(y, M)
        
    return y
def up_sample(x, L):
    y = np.zeros(L*len(x))
    y[::L] = x
    return y

def interpolate(x, L,filter_type):
    y=up_sample(x,L)
    if filter_type == 'shanon':
        h = L * signal.firwin(1024, cutoff=1/L, window='hamming')
        y=signal.convolve(y, h, mode='same')
    elif filter_type == 'ZOH':
        h=np.ones(L)
        y=signal.convolve(y, h, mode='same')
    elif filter_type == 'FOH':
        h=np.ones(L)
        h=signal.convolve(h, h, mode='full')
        y=signal.convolve(y, h, mode='same')
    return y