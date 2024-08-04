import numpy as np
import scipy
from scipy.io.wavfile import write
import io

def read_pdm_file(file_path):
    """
    Reads a PDM signal from a text file. Assumes the signal is composed of -1 and 1 values.

    Parameters:
        file_path (str): Path to the PDM file.

    Returns:
        np.array: Array containing the PDM signal.
    """
    pdm_signal = []

    # Open the file and read each line
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into individual values and append them to the signal list
            values = line.split()
            for val in values:
                pdm_signal.append(int(val))

    # Convert the list to a numpy array of type int8
    pdm_array = np.array(pdm_signal, dtype=np.int8)

    return pdm_array

def moving_average(signal, window_size):
    """
    Computes the moving average of the input signal using a specified window size.

    Parameters:
        signal (np.array): The input signal.
        window_size (int): The window size for the moving average.

    Returns:
        np.array: The signal after applying the moving average.
    """
    # Pad the signal with zeros at the beginning
    pad_width = window_size - 1
    padding = np.zeros(pad_width, dtype=signal.dtype)
    padded_signal = np.concatenate((padding, signal), axis=0)

    # Compute the cumulative sum of the padded signal
    cumsum = np.cumsum(padded_signal)

    # Calculate the moving average using the cumulative sum
    window_size_float = float(window_size)
    moving_sum = cumsum[window_size:] - cumsum[:-window_size]
    moving_average_result = moving_sum / window_size_float

    return moving_average_result

def cic_filter(pdm_signal, decimation_factor=64, order=1):
    """
    Applies a CIC filter to the PDM signal.

    Parameters:
        pdm_signal (np.array): The PDM signal to filter.
        decimation_factor (int): The decimation factor for the CIC filter.
        order (int): The order of the CIC filter.

    Returns:
        np.array: The filtered PCM signal.
    """
    # Step 1: Integrator stage (Moving Average)
    integrator = pdm_signal
    for _ in range(order):
        integrator = moving_average(integrator, decimation_factor)

    # Step 2: Decimation
    decimated_integrator = integrator[::decimation_factor]

    # Step 3: Comb stage
    comb = np.diff(decimated_integrator)

    return comb

def save_pcm_as_wav_file(pcm_signal, sample_rate):
    """
    Saves the PCM signal as a bytes object.

    Parameters:
        pcm_signal (np.array): The PCM signal to save. This should be a 1D numpy array of raw PCM data.
        sample_rate (int): The sample rate of the PCM signal, in Hz (samples per second).

    Returns:
        bytes: The WAV file data as a bytes object.
    """
    # Normalize the PCM signal
    max_val = np.max(np.abs(pcm_signal))
    normalized_signal = pcm_signal / max_val
    scaled_signal = normalized_signal * 32767
    int16_signal = np.int16(scaled_signal)

    # Save the PCM signal to a WAV file in memory
    wav_buffer = io.BytesIO()
    write(wav_buffer, sample_rate, int16_signal)
    wav_data = wav_buffer.getvalue()
    return wav_data

def Pdm2Pcm(pdm_file_path):
    """
    Converts a PDM signal from a file to a PCM signal and saves it as a WAV file.

    Parameters:
        pdm_file_path (str): Path to the PDM file.

    Returns:
        bytes: The WAV file data as a bytes object.
    """

    # Read PDM signal from file
    pdm_signal = read_pdm_file(pdm_file_path)

    # Convert PDM signal to PCM signal
    pcm_signal = cic_filter(pdm_signal, decimation_factor, order)

    # Save the PCM signal as a variable
    wav_file = save_pcm_as_wav_file(pcm_signal, pcm_sample_rate)

    return wav_file
