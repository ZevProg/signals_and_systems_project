import wave
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def real_time_filter(input_file, output_file, cutoff_frequency=100, numtaps=4400):
    def sinc_high_pass_filter(cutoff, fs, numtaps):
        t = np.arange(numtaps) - (numtaps - 1) / 2
        sinc_func = np.sinc(2 * cutoff * t / fs)
        window = np.hamming(numtaps)
        sinc_func *= window
        sinc_func /= np.sum(sinc_func)
        delta = np.zeros(numtaps)
        delta[(numtaps - 1) // 2] = 1
        return delta - sinc_func

    # Read the WAV file
    fs, signal = wavfile.read(input_file)

    # Convert to float32 for processing
    signal = signal.astype(np.float32)

    # If stereo, convert to mono by averaging channels
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    # Normalize the signal
    signal = signal / np.max(np.abs(signal))

    # Get the high-pass filter coefficients
    high_pass_filter = sinc_high_pass_filter(cutoff_frequency, fs, numtaps)

    # Initialize output signal
    output_signal = np.zeros_like(signal)
    # Buffer to store previous input values (length of filter coefficients)
    buffer = np.zeros(numtaps)

    # Apply the filter sample-by-sample
    for n in range(len(signal)):
        # Update the buffer with the current input sample
        buffer = np.roll(buffer, 1)
        buffer[0] = signal[n]

        # Compute the output sample by convolving the filter coefficients with the buffer
        output_signal[n] = np.dot(high_pass_filter, buffer)

    # Save the filtered signal as a new WAV file
    wavfile.write(output_file, fs, (output_signal * 32767).astype(np.int16))

# Usage example:
real_time_filter('aspose_פסנתר-+ביחד.wav', 'output_filtered.wav')
