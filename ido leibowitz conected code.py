#ido leibowitz
"""
conecting decimation and interpolation
to transmission and recption ssb
"""

from random import sample
import numpy as np
import scipy.signal as signal
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import scipy
from scipy.io import wavfile
import os
import threading

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
    padded_x = np.pad(x, (m - 1, m - 1), 'constant')
    flip_y = np.flip(y)
    result_length = n + m - 1
    result = np.zeros(result_length)
    for i in range(result_length):
        result[i] = np.sum(padded_x[i:i + m] * flip_y)
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
    normalized_signal = np.int16(
        downsampled_signal / np.max(np.abs(downsampled_signal)) * 32767)  # Normalize signal to int16 range

    output_filename = generate_new_filename_decimation(input_filename)
    wavfile.write(output_filename, new_samplerate, normalized_signal)

    return output_filename


def my_interpolation_LPF(L, LPF_type):
    '''
    L: interpolation factor
    n: number of samples
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
        h = np.ones(L)
        h = my_convolution(h, h, mode='full')
        h = h / np.max(h)
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





# Load the WAV file
def load_wav(filename):
    data, samplerate = sf.read(filename)
    if data.ndim > 1:  # Handle stereo by taking only one channel
        data = data[:, 0]  # Use only the first channel for stereo files
    return data, samplerate


# Save the WAV file
def save_wav(filename, data, samplerate):
    sf.write(filename, data, samplerate)


# Generate a SSB signal
def ssb_modulate(signal, carrier_freq, samplerate):
    t = np.arange(len(signal)) / samplerate
    signal_mul_cos = signal * np.cos(2 * np.pi * carrier_freq * t)  # Multiply with cosine
    signal_hilbert = scipy.signal.hilbert(signal)  # Apply Hilbert transform
    signal_mul_sin = np.imag(signal_hilbert) * np.sin(2 * np.pi * carrier_freq * t)  # Multiply with sine
    ssb_signal = signal_mul_cos - signal_mul_sin  # SSB signal (Upper Sideband)
    return ssb_signal


# Low-pass filter using FFT
def low_pass_filter(signal, cutoff_freq, samplerate):
    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1 / samplerate)
    filtered_fft_signal = fft_signal.copy()
    filtered_fft_signal[np.abs(freqs) > cutoff_freq] = 0  # Zero out frequencies above cutoff
    filtered_signal = 2 * np.fft.ifft(filtered_fft_signal)
    return np.real(filtered_signal)


# Demodulate the SSB signal
def ssb_demodulate(ssb_signal, carrier_freq, samplerate):
    t = np.arange(len(ssb_signal)) / samplerate
    demodulated_signal = ssb_signal * np.cos(2 * np.pi * carrier_freq * t)  # Multiply with cosine
    cutoff_freq = 4000  # Low-pass filter cutoff frequency
    recovered_signal = low_pass_filter(demodulated_signal, cutoff_freq, samplerate)  # Apply low-pass filter
    return recovered_signal


# Real-time recording and processing callback
def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(status, flush=True)  # Print status if there's an error

    ssb_signal = ssb_modulate(indata[:, 0], carrier_freq, samplerate)  # Modulate input signal
    recovered_signal = ssb_demodulate(ssb_signal, carrier_freq, samplerate)  # Demodulate signal

    if outdata is not None:
        outdata[:, 0] = recovered_signal  # Output recovered signal to both channels
        outdata[:, 1] = recovered_signal


# Main function for processing file or live input
def process_audio(mode='file', filename=None, carrier_freq=10000):
    global samplerate
    if mode == 'file' and filename:
        # Load the input WAV file
        data, samplerate = load_wav(filename)
        carrier_freq = samplerate / 2  # Set carrier frequency to half the sample rate

        # Modulate the signal using SSB
        ssb_signal = ssb_modulate(data, carrier_freq, samplerate)

        # Demodulate the signal to recover the original data
        recovered_signal = 0.5 * ssb_demodulate(ssb_signal, carrier_freq, samplerate)

        # Save the recovered signal to an output WAV file
        output_filename = 'output_test_' + filename
        save_wav(output_filename, recovered_signal, samplerate)

        # Plotting the signals
        t = np.arange(len(data)) / samplerate

        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(np.fft.fftfreq(len(t)), np.fft.fft(data))
        plt.title('Original Baseband Signal')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')

        plt.subplot(3, 1, 2)
        plt.plot(np.fft.fftfreq(len(t)), np.fft.fft(ssb_signal))
        plt.title('SSB Modulated Signal (Upper Sideband)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')

        plt.subplot(3, 1, 3)
        plt.plot(np.fft.fftfreq(len(t)), np.fft.fft(recovered_signal))
        plt.title('Demodulated Signal')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')

        plt.tight_layout()
        plt.show()

    elif mode == 'live':
        global recorded_data
        recorded_data = []

        # Thread for stopping the recording on Enter key press
        def wait_for_enter():
            input("Press Enter to stop recording...\n")
            sd.stop()

        enter_thread = threading.Thread(target=wait_for_enter)
        enter_thread.start()

        def callback(indata, outdata, frames, time, status):
            recorded_data.append(indata.copy())
            audio_callback(indata, outdata, frames, time, status)

        # Start the stream for real-time recording and processing
        with sd.Stream(samplerate=samplerate, channels=2, callback=callback):
            print("Recording and processing in real-time. Press Enter to stop.")
            enter_thread.join()  # Wait until Enter is pressed

        # Convert the list of arrays to a single array
        recorded_data = np.concatenate(recorded_data, axis=0)

        # Process the recorded data for plotting purposes
        t = np.arange(len(recorded_data)) / samplerate
        save_wav('recording.wav', recorded_data, samplerate)
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(np.fft.fftfreq(len(t), 1 / samplerate), np.abs(np.fft.fft(recorded_data[:, 0])))
        plt.title('Original Baseband Signal')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')

        ssb_signal = ssb_modulate(recorded_data[:, 0], carrier_freq, samplerate)
        plt.subplot(3, 1, 2)
        plt.plot(np.fft.fftfreq(len(t), 1 / samplerate), np.abs(np.fft.fft(ssb_signal)))
        plt.title('SSB Modulated Signal (Upper Sideband)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')

        recovered_signal = ssb_demodulate(ssb_signal, carrier_freq, samplerate)
        save_wav('recording_modulated.wav', recovered_signal, samplerate)
        plt.subplot(3, 1, 3)
        plt.plot(np.fft.fftfreq(len(t), 1 / samplerate), np.abs(np.fft.fft(recovered_signal)))
        plt.title('Demodulated Signal')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')

        plt.tight_layout()
        plt.show()
    else:
        print("Invalid mode or filename not provided for file mode.")





# Check the functions
M = 4
input_filename1 = "about_time.wav"
decimated_signal1 = decimate(input_filename1, M)
M = 4
input_filename2 = "activity_unproductive.wav"
decimated_signal2 = decimate(input_filename2, M)
L = 5
filter_type = 'shanon'
input_filename3 = "about_time.wav"
interpolate_signal = interpolate(input_filename3, L, filter_type)


# Set the sample rate and carrier frequency
samplerate = 44100
carrier_freq = 10000

process_audio(mode='file',filename='about_time_new_deci.wav')
process_audio(mode='file',filename='about_time_new_inter.wav')
# Run in live mode
process_audio(mode='live', carrier_freq=carrier_freq)

