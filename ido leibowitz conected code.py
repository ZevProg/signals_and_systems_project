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
    y= np.int16(y/np.max(np.abs(y))*32767)
    wavfile.write(output_filename, samplerate, y)

    return y


def my_interpolation_LPF(L, LPF_type):
    '''
    L: interpolation factor
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


def up_sample(x, L):
    '''
    x: input signal
    L: interpolation factor
    return: upsampled signal
    '''
    y = np.zeros(L * len(x))
    y[::L] = x
    return y


def interpolate(input_filename, L, output_filename, filter_type):
    '''
    x: input signal
    L: interpolation factor
    filter_type: 'shanon', 'ZOH', 'FOH'
    return: upsampled signal include interpolation filter
    '''

    samplerate, x = wavfile.read(input_filename)
    y = up_sample(x, L)
    h = my_interpolation_LPF(L, filter_type)
    y = my_convolution(y, h, mode='same')

    samplerate = int(samplerate * L)
    y= np.int16(y/np.max(np.abs(y))*32767)
    wavfile.write(output_filename, samplerate, y)

    return y




# Load the WAV file
def load_wav(filename):
    data, samplerate = sf.read(filename)
    if data.ndim > 1:  # Handle stereo by taking only one channel
        data = data[:, 0]
    return data, samplerate

# Save the WAV file
def save_wav(filename, data, samplerate):
    sf.write(filename, data, samplerate)

# Generate a SSB signal
def ssb_modulate(signal, carrier_freq, samplerate):
    t = np.arange(len(signal)) / samplerate
    ## cosine part
    signa_mul_cos = signal * np.cos(2 * np.pi * carrier_freq * t)
    ## hilbert transform
    signal_filter_hilbert = scipy.signal.hilbert(signal)
    siganal_mul_sin = np.real(signal_filter_hilbert) * np.sin(2 * np.pi * carrier_freq * t)
    ssb_signal = signa_mul_cos - siganal_mul_sin ## SSB signal - upper sideband
    return ssb_signal

# Demodulate the SSB signal
def ssb_demodulate(ssb_signal, carrier_freq, samplerate):
    t = np.arange(len(ssb_signal)) / samplerate
    demodulated_signal = ssb_signal * np.cos(2 * np.pi * carrier_freq * t)
    # Apply a low-pass filter to recover the original signal
    nyquist_rate = samplerate / 2.0
    cutoff = 4000  # Low-pass filter cutoff frequency
    b, a = signal.butter(5, cutoff / nyquist_rate)
    recovered_signal = signal.lfilter(b, a, demodulated_signal)
    return recovered_signal

# Real-time recording and processing callback
def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(status, flush=True)

    ssb_signal = ssb_modulate(indata[:, 0], carrier_freq, samplerate)
    recovered_signal = ssb_demodulate(ssb_signal, carrier_freq, samplerate)

    outdata[:, 0] = recovered_signal
    outdata[:, 1] = recovered_signal

# the transmission and recption ssb function
def process_audio(mode='file', filename=None, duration=5, carrier_freq=10000):
    global samplerate
    if mode == 'file' and filename:
        # Load the input WAV file
        data, samplerate = load_wav(filename)
        carrier_freq = samplerate/2

        # Modulate the signal using SSB
        ssb_signal = ssb_modulate(data, carrier_freq, samplerate)

        # Demodulate the signal to recover the original data
        recovered_signal = ssb_demodulate(ssb_signal, carrier_freq, samplerate)

        # Save the recovered signal to an output WAV file
        output_filename = 'output_test_' + filename
        save_wav(output_filename, recovered_signal, samplerate)

        # Plotting
        t = np.arange(len(data)) / samplerate

        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(np.fft.fftfreq(len(t)), np.fft.fft(data))
        plt.title('Original Baseband Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.subplot(3, 1, 2)
        plt.plot(np.fft.fftfreq(len(t)), np.fft.fft(ssb_signal))
        plt.title('SSB Modulated Signal (Upper Sideband)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.subplot(3, 1, 3)
        plt.plot(np.fft.fftfreq(len(t)), np.fft.fft(recovered_signal))
        plt.title('Demodulated Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.tight_layout()
        plt.show()

    elif mode == 'live':
        # Start the stream for real-time recording and processing
        with sd.Stream(samplerate=samplerate, channels=2, callback=audio_callback):
            print("Recording and processing in real-time. Press Ctrl+C to stop.")
            sd.sleep(int(duration * 1000))

        # Record a bit of data for plotting purposes
        print("Recording for plotting purposes...")
        recorded_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
        sd.wait()

        t = np.arange(len(recorded_data)) / samplerate
        save_wav('recording.wav',recorded_data,samplerate)
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(t, recorded_data)
        plt.title('Original Baseband Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        ssb_signal = ssb_modulate(recorded_data[:, 0], carrier_freq, samplerate)
        plt.subplot(3, 1, 2)
        plt.plot(t, ssb_signal)
        plt.title('SSB Modulated Signal (Upper Sideband)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        recovered_signal = ssb_demodulate(ssb_signal, carrier_freq, samplerate)
        save_wav('recording_modulated.wav',recovered_signal,samplerate)
        plt.subplot(3, 1, 3)
        plt.plot(t, recovered_signal)
        plt.title('Demodulated Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.tight_layout()
        plt.show()
    else:
        print("Invalid mode or filename not provided for file mode.")


# Check the functions
M = 4
input_filename = "about_time.wav"
output_filename = "deci_about_time_M_" + str(M) + ".wav"
decimated_signal = decimate(input_filename, M, output_filename)

L = 5
filter_type = 'ZOH'
input_filename = "about_time.wav"
output_filename = "inter_about_time_L_" + str(L) + ".wav"
interpolate_signal = interpolate(input_filename, L, output_filename, filter_type)

# Set the sample rate and carrier frequency
samplerate = 44100
carrier_freq = 10000

process_audio(mode='file',filename='deci_about_time_M_4.wav')
#process_audio(mode='live', duration=10, carrier_freq=carrier_freq)

