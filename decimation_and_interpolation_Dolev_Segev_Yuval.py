import numpy as np
from scipy.io import wavfile

def my_convolution(x, y, mode='full'):
    '''
    x: input signal
    y: impulse response
    mode: 'full' or 'same' 
    full: returns the convolution of x and y
    same: returns the central part of the convolution of x and y
    return: convolution of x and y 
    '''
    n = len(x)
    m = len(y)
    # Padding the signal for multiplication
    padded_x = np.pad(x, (m-1, m-1), 'constant')
    flip_y = np.flip(y)
    result_length = n + m - 1
    result = np.zeros(result_length)
    for i in range(result_length):
        result[i] = np.sum(padded_x[i:i+m] * flip_y)
    if mode == 'full':
        return result
    elif mode == 'same':
        start = (m - 1) // 2
        return result[start:start + n]
    return result

def down_sample(x, M):
    '''
    x: input signal
    M: decimation factor
    return: downsampled signal
    '''
    return x[::M]

def decimate(input_filename, M, output_filename):
    '''
    x: input signal
    M: decimation factor
    return: downsampled signal include anti-aliasing filter 
    '''
    samplerate, x = wavfile.read(input_filename)
    
    n = np.arange(-20 * M, 20 * M + 1)
    h = np.sin(np.pi / M * n) / (np.pi * n)
    h[n == 0] = 1 / M
    h /= np.sum(h)  # Normalize filter coefficients
    
    y = my_convolution(x, h, mode='same')
    y = down_sample(y, M)
    
    samplerate = int(samplerate / M)
    y = y / np.max(np.abs(y))   
    wavfile.write(output_filename, samplerate, y)
    
    return y

def my_interpolation_LPF(L,LPF_type):
    '''
    L: interpolation factor
    LPF_type: 'shanon', 'ZOH', 'FOH'
    return: impulse response of the interpolation filter
    '''
    if LPF_type=='shanon':
        n=np.arange(- 20*L,20*L+1)
        h=L*np.sin(np.pi/L*n)/(np.pi*n)   
        h[n==0]=1 
    elif LPF_type=='ZOH':
        h=np.ones(L)
    elif LPF_type=='FOH':
        h=np.ones(L)
        h=my_convolution(h, h, mode='full')
        h=h/np.max(h)
    return h

def up_sample(x, L):
    '''
    x: input signal
    L: interpolation factor
    return: upsampled signal
    '''
    y = np.zeros(L*len(x))
    y[::L] = x
    return y

def interpolate(input_filename, L, output_filename,filter_type):
    '''
    x: input signal
    L: interpolation factor
    filter_type: 'shanon', 'ZOH', 'FOH'
    return: upsampled signal include interpolation filter
    '''
    
    samplerate, x = wavfile.read(input_filename)
    y=up_sample(x,L)
    h=my_interpolation_LPF(L,filter_type)
    y=my_convolution(y, h, mode='same')
    
    samplerate = int(samplerate * L)
    y = y / np.max(np.abs(y))  
    wavfile.write(output_filename, samplerate, y)
     
    return y


# Check the functions
M = 4
input_filename = "about_time.wav"
output_filename = "deci_about_time_M_"+str(M)+".wav"
decimated_signal = decimate(input_filename, M, output_filename)

L = 5
filter_type = 'shanon'
input_filename = "about_time.wav"
output_filename = "inter_about_time_L_"+str(L)+".wav"
interpolate_signal = interpolate(input_filename, L, output_filename, filter_type)

