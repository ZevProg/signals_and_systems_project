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
    with open(file_path, 'r') as file:
        pdm_signal = file.read().split()
    return np.array([int(val) for val in pdm_signal], dtype=np.int8)


def cic_filter(pdm_signal, decimation_factor=32):
    """
    Applies a CIC filter to the PDM signal.

    Parameters:
        pdm_signal (np.array): The PDM signal to filter.
        decimation_factor (int): The decimation factor for the CIC filter.

    Returns:
        np.array: The filtered PCM signal.
    """
    integrator = np.cumsum(pdm_signal)
    comb = np.diff(integrator[::decimation_factor])
    return comb


def pdm_to_pcm(pdm_signal, decimation_factor=32):
    """
    Converts a PDM signal to a PCM signal using a CIC filter.

    Parameters:
        pdm_signal (np.array): The PDM signal to convert.
        decimation_factor (int): The decimation factor for the CIC filter.

    Returns:
        np.array: The PCM signal.
    """
    pcm_signal = cic_filter(pdm_signal, decimation_factor)
    return pcm_signal


def save_pcm_as_wav(pcm_signal, sample_rate, file_path):
    """
    Saves the PCM signal as a WAV file.

    Parameters:
        pcm_signal (np.array): The PCM signal to save.
        sample_rate (int): The sample rate of the PCM signal.
        file_path (str): The path to save the WAV file.
    """
    pcm_signal = np.int16(pcm_signal / np.max(np.abs(pcm_signal)) * 32767)
    write(file_path, sample_rate, pcm_signal)


def main():
    # Parameters
    pdm_file_path = r'C:\Users\User\Documents\otot_project\Counting.txt'
    wav_file_path = r'C:\Users\User\Documents\otot_project1\output.wav'
    sample_rate =307200   # Desired sample rate for the WAV file
    decimation_factor = 32  # Decimation factor for the CIC filter

    # Read PDM signal from file
    pdm_signal = read_pdm_file(pdm_file_path)
    print(f"First 10 samples of PDM signal: {pdm_signal[:10]}")

    # Convert PDM signal to PCM signal
    pcm_signal = pdm_to_pcm(pdm_signal, decimation_factor)
    print(f"First 10 samples of PCM signal: {pcm_signal[:10]}")

    # Save the PCM signal as a WAV
    save_pcm_as_wav(pcm_signal, sample_rate // decimation_factor, wav_file_path)
    print(f"WAV file saved at: {wav_file_path}")


if __name__ == "__main__":
    main()
