import numpy as np
from scipy.io import wavfile
import os

def my_convolution(signal, impulse_response, mode='full'):
    '''
    signal: input signal
    impulse_response: impulse response
    mode: 'full' or 'same'
    full: returns the convolution of signal and impulse_response
    same: returns the central part of the convolution of signal and impulse_response
    return: convolution of signal and impulse_response
    '''
    signal_length = len(signal)
    impulse_length = len(impulse_response)
    # Padding the signal for multiplication
    padded_signal = np.pad(signal, (impulse_length - 1, impulse_length - 1), 'constant')
    flip_impulse_response = np.flip(impulse_response)
    result_length = signal_length + impulse_length - 1
    convolution_result = np.zeros(result_length)
    for i in range(result_length):
        convolution_result[i] = np.sum(padded_signal[i:i + impulse_length] * flip_impulse_response)
    if mode == 'full':
        return convolution_result
    elif mode == 'same':
        start_index = (impulse_length - 1) // 2
        return convolution_result[start_index:start_index + signal_length]
    return convolution_result


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
    normalized_signal = np.int16(
        downsampled_signal / np.max(np.abs(downsampled_signal)) * 32767)  # Normalize signal to int16 range

    output_filename = generate_new_filename_decimation(input_filename)
    wavfile.write(output_filename, new_samplerate, normalized_signal)

    return output_filename


def my_interpolation_LPF(L, LPF_type):
    '''
    L: interpolation factor
    n: number of samples
    h: impulse response of the interpolation filter
    LPF_type: 'shanon', 'ZOH', 'FOH'
    return: impulse response of the interpolation filter
    '''
    if LPF_type == 'shanon':
        n = np.arange(- 20 * L, 20 * L + 1)
        h = L * np.sin(np.pi / L * n) / (np.pi * n)
        h[n == 0] = 1
    elif LPF_type == 'ZOH':
        h = np.ones(L)
    elif LPF_type == 'FOH':
        impuls_response_step = np.ones(L)
        impuls_response_triangle = my_convolution( impuls_response_step,  impuls_response_step, mode='full')
        h = impuls_response_triangle / np.max(impuls_response_triangle)
    return h


def up_sample(input_signal, interpolation_factor):
    up_sampled_signal = np.zeros(interpolation_factor * len(input_signal))
    up_sampled_signal[::interpolation_factor] = input_signal
    return up_sampled_signal


def interpolate(input_filename, L, filter_type):
    '''
    x: input signal
    L: interpolation factor
    filter_type: 'shanon', 'ZOH', 'FOH'
    return:new wav file with upsampled signal include interpolation filter
    '''

    samplerate, signals_sample = wavfile.read(input_filename)
    upsampled_signal = up_sample(signals_sample, L)
    interpolation_filter = my_interpolation_LPF(L, filter_type)
    filtered_signal = my_convolution(upsampled_signal, interpolation_filter, mode='same')

    new_samplerate = int(samplerate * L)
    normalized_signal = np.int16(filtered_signal / np.max(np.abs(filtered_signal)) * 32767)
    output_filename = generate_new_filename_interpolation(input_filename)
    wavfile.write(output_filename, new_samplerate, normalized_signal)

    return output_filename