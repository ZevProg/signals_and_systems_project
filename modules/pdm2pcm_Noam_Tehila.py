import numpy as np
from scipy.io.wavfile import write

def read_pdm_file(file_path):
    """
    Reads a PDM signal from a text file. Assumes the signal is composed of -1 and 1 values.

    Parameters:
        file_path (str): Path to the PDM file.

    Returns:
        np.array: Array containing the PDM signal.
    """
    pdm_signal = []

    with open(file_path, 'r') as file:
        for line in file:
            values = line.split()
            for val in values:
                pdm_signal.append(int(val))

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
    cumsum = np.cumsum(np.insert(signal, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)


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
    comb = decimated_integrator
    for _ in range(order):
        comb = np.diff(comb)

    return comb


def pdm_to_pcm(pdm_signal, decimation_factor=64, order=1):
    """
    Converts a PDM signal to a PCM signal using a CIC filter.

    Parameters:
        pdm_signal (np.array): The PDM signal to convert.
        decimation_factor (int): The decimation factor for the CIC filter.
        order (int): The order of the CIC filter.

    Returns:
        np.array: The PCM signal.
    """
    pcm_signal = cic_filter(pdm_signal, decimation_factor, order)
    return pcm_signal


def save_pcm_as_wav(pcm_signal, sample_rate, file_path):
    """
    Saves the PCM signal as a WAV file.

    Parameters:
        pcm_signal (np.array): The PCM signal to save. This should be a 1D numpy array of raw PCM data.
        sample_rate (int): The sample rate of the PCM signal, in Hz (samples per second).
        file_path (str): The path (including the filename) where the WAV file will be saved.

    Returns:
        None
    """
    max_val = np.max(np.abs(pcm_signal))
    normalized_signal = pcm_signal / max_val
    scaled_signal = normalized_signal * 32767
    int16_signal = np.int16(scaled_signal)
    write(file_path, sample_rate, int16_signal)


def main():
    # Parameters
    pdm_file_path = r'C:\Users\User\Documents\otot_project\Counting.txt'
    wav_file_path = r'C:\Users\User\Documents\otot_project\output.wav'
    pdm_sample_rate = 3072000  # Sample rate of the PDM file
    pcm_sample_rate = 8000  # Desired sample rate for the WAV file
    decimation_factor = 64  # Decimation factor for the CIC filter
    order = 5  # Order of the CIC filter

    # Read PDM signal from file
    pdm_signal = read_pdm_file(pdm_file_path)
    print(f"First 10 samples of PDM signal: {pdm_signal[:10]}")

    # Convert PDM signal to PCM signal
    pcm_signal = pdm_to_pcm(pdm_signal, decimation_factor, order)
    print(f"First 10 samples of PCM signal: {pcm_signal[:10]}")

    # Save the PCM signal as a WAV
    save_pcm_as_wav(pcm_signal, pcm_sample_rate, wav_file_path)
    print(f"WAV file saved at: {wav_file_path}")


if __name__ == "__main__":
    main()
