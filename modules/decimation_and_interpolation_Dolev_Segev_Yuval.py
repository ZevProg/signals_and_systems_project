import numpy as np
from scipy.io import wavfile
import os

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

def generate_new_filename_decimation(input_filename):
    '''
    input_filename: The original input file name
    return: New filename with "new" inserted before the file extension
    '''
    base, ext = os.path.splitext(input_filename)
    new_filename = f"{base}_new_deci{ext}"
    return new_filename

def generate_new_filename_interpolation(input_filename):
    '''
    input_filename: The original input file name
    return: New filename with "new" inserted before the file extension
    '''
    base, ext = os.path.splitext(input_filename)
    new_filename = f"{base}_new_inter{ext}"
    return new_filename

    
def down_sample(filtered_signal, decimation_factor):
    '''
    x: input signal
    M: decimation factor
    return: downsampled signal
    '''
    return filtered_signal[::decimation_factor]

def decimate(input_filename, decimation_factor):
  
    samplerate, input_signal = wavfile.read(input_filename)
    
    time_indices = np.arange(-20 * decimation_factor, 20 * decimation_factor + 1)
    sinc_filter = np.sin(np.pi / decimation_factor * time_indices) / (np.pi * time_indices)
    sinc_filter[time_indices == 0] = 1 / decimation_factor

    filtered_signal = my_convolution(input_signal, sinc_filter, mode='same')
    downsampled_signal = down_sample(filtered_signal, decimation_factor)
    
    new_samplerate = int(samplerate / decimation_factor)
    normalized_signal = np.int16(downsampled_signal / np.max(np.abs(downsampled_signal)) * 32767)  # Normalize signal to int16 range
    
    output_filename = generate_new_filename_decimation(input_filename)
    wavfile.write(output_filename, new_samplerate, normalized_signal)
    
    return output_filename

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

def up_sample(input_signal, interpolation_factor):
    
    up_sampled_signal = np.zeros(interpolation_factor*len(input_signal))
    up_sampled_signal[::interpolation_factor] = input_signal
    return up_sampled_signal

def interpolate(input_filename, L ,filter_type):
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
    
    new_samplerate = int(samplerate * L)
    y = np.int16(y / np.max(np.abs(y)) * 32767) 
    output_filename = generate_new_filename_interpolation(input_filename)
    wavfile.write(output_filename, new_samplerate, y)
     
    return output_filename


# Check the functions
M = 4
input_filename = "about_time.wav"
decimated_signal = decimate(input_filename, M)

L = 5
filter_type = 'shanon'
input_filename = "about_time.wav"
interpolate_signal = interpolate(input_filename, L, filter_type)

