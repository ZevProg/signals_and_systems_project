import numpy as np
import matplotlib.pyplot as plt
# Create Sinc-based high-pass filter
def sinc_high_pass_filter(cutoff, fs, numtaps):
    """
    Create a high-pass filter using the sinc function.

    Parameters:
    cutoff (float): Cutoff frequency in Hz
    fs (float): Sampling frequency in Hz
    numtaps (int): Number of filter taps

    Returns:
    numpy array: High-pass filter coefficients
    """
    # Generate the time vector centered around zero
    t = np.arange(numtaps) - (numtaps - 1) / 2
    # Create sinc function for the low-pass filter
    sinc_func = np.sinc(2 * cutoff * t / fs)
    # Apply a window function to the sinc function
    window = np.hamming(numtaps)
    sinc_func *= window
    # Normalize the filter coefficients
    sinc_func /= np.sum(sinc_func)
    # Create high-pass filter by subtracting from delta function
    delta = np.zeros(numtaps)
    delta[(numtaps - 1) // 2] = 1
    hpf = delta - sinc_func
    return hpf

# Parameters
cutoff_frequency = 100  # Cutoff frequency in Hz
numtaps = 4400  # Number of taps in the filter


# Real-time filtering function using discrete convolution
def real_time_filter(input_signal, filter_coefficients):
    """
    Apply a real-time FIR filter using discrete convolution.

    Parameters:
    input_signal (numpy array): The input signal to be filtered.
    filter_coefficients (numpy array): The FIR filter coefficients.

    Returns:
    numpy array: The filtered output signal.
    """
    # Initialize output signal
    output_signal = np.zeros_like(input_signal)
    # Buffer to store previous input values (length of filter coefficients)
    buffer = np.zeros(len(filter_coefficients))

    # Apply the filter sample-by-sample
    for n in range(len(input_signal)):
        # Update the buffer with the current input sample
        buffer = np.roll(buffer, 1)
        buffer[0] = input_signal[n]

        # Compute the output sample by convolving the filter coefficients with the buffer
        output_signal[n] = np.dot(filter_coefficients, buffer)

    return output_signal