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
    pdm_signal = []  # Initialize an empty list to store the PDM signal

    # Open the file for reading
    with open(file_path, 'r') as file:
        # Read the file line by line
        for line in file:
            # Split the line into individual values
            values = line.split()
            # Convert each value to an integer and append to the pdm_signal list
            for val in values:
                pdm_signal.append(int(val))

    # Convert the list to a numpy array with data type int8
    pdm_array = np.array(pdm_signal, dtype=np.int8)

    return pdm_array


def cic_filter(pdm_signal, decimation_factor=32):
    """
    Applies a CIC filter to the PDM signal.

    Parameters:
        pdm_signal (np.array): The PDM signal to filter.
        decimation_factor (int): The decimation factor for the CIC filter.

    Returns:
        np.array: The filtered PCM signal.
    """
    # Step 1: Integrator stage
    # Compute the cumulative sum of the PDM signal
    integrator = np.cumsum(pdm_signal)

    # Step 2: Decimation
    # Decimate the integrator output by taking every 'decimation_factor'-th sample
    decimated_integrator = integrator[::decimation_factor]

    # Step 3: Comb stage
    # Compute the difference between consecutive samples in the decimated integrator output
    comb = np.diff(decimated_integrator)

    # Return the comb output as the filtered PCM signal
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
        pcm_signal (np.array): The PCM signal to save. This should be a 1D numpy array of raw PCM data.
        sample_rate (int): The sample rate of the PCM signal, in Hz (samples per second).
        file_path (str): The path (including the filename) where the WAV file will be saved.

    Returns:
        None
    """
    # Normalize the PCM signal to the range of int16
    # np.max(np.abs(pcm_signal)) finds the maximum absolute value in the signal
    # This ensures the signal is scaled between -1 and 1
    max_val = np.max(np.abs(pcm_signal))
    normalized_signal = pcm_signal / max_val

    # Scale the normalized signal to the int16 range
    scaled_signal = normalized_signal * 32767

    # Convert the scaled signal to int16 data type
    int16_signal = np.int16(scaled_signal)

    # Write the int16 PCM data to a WAV file
    write(file_path, sample_rate, int16_signal)



def main():
    # Parameters
    pdm_file_path = r'C:\Users\User\Documents\otot_project\Counting.txt'
    wav_file_path = r'C:\Users\User\Documents\otot_project\output_counting.wav'
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
