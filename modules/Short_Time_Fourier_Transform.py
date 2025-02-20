import numpy as np  # Import numpy for mathematical computations
from scipy.io import wavfile  # Import wavfile module for reading and writing WAV files
import warnings  # Import warnings module for managing warnings

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)  # Ignore WAV warnings

# Function to create a Hanning window
# Input: window_length (int) - the length of the window
# Output: A numpy array representing the Hanning window
def CreateHanningWindow(window_length):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(window_length) / (window_length - 1)))  # Create Hanning window of given length

# Function to compute the Fast Fourier Transform (FFT)
# Input: signal (numpy array) - the input signal
# Output: A numpy array representing the FFT of the input signal
def FFT(signal):
    N = len(signal)  # Length of the signal
    if N <= 1:  # If the signal length is 1 or less
        return signal  # Return the signal as it is
    even = FFT(signal[0::2])  # Compute FFT for even indexed elements
    odd = FFT(signal[1::2])  # Compute FFT for odd indexed elements
    T = np.exp(-2j * np.pi * np.arange(N) / N)  # Compute FFT factors
    return np.concatenate([even + T[:N // 2] * odd, even + T[N // 2:] * odd])  # Combine results

# Function to compute the Inverse Fast Fourier Transform (IFFT)
# Input: spectrum (numpy array) - the input spectrum
# Output: A numpy array representing the IFFT of the input spectrum
def IFFT(spectrum):
    N = len(spectrum)  # Length of the spectrum
    if N <= 1:  # If the spectrum length is 1 or less
        return spectrum  # Return the spectrum as it is
    even = IFFT(spectrum[0::2])  # Compute IFFT for even indexed elements
    odd = IFFT(spectrum[1::2])  # Compute IFFT for odd indexed elements
    T = np.exp(2j * np.pi * np.arange(N) / N)  # Compute IFFT factors
    return (np.concatenate([even + T[:N // 2] * odd, even + T[N // 2:] * odd]) / 2)  # Combine results and divide by 2

# Function to compute the Short-Time Fourier Transform (STFT)
# Input: input_wav (str) - path to the input WAV file
#        window_size (int) - the size of the window (default 2048)
#        hop_size (int) - the hop size (default 512)
# Output: A tuple containing the STFT matrix and the sample rate
def STFT(input_wav, window_size=2048, hop_size=512):
    sample_rate, audio_signal = wavfile.read(input_wav)  # Read WAV file and get sample rate and signal

    audio_signal = audio_signal.astype(np.float64)  # Convert to float64
    if np.issubdtype(audio_signal.dtype, np.integer):  # Check if the signal is of integer type
        audio_signal /= np.iinfo(audio_signal.dtype).max  # Normalize by the maximum possible value for the data type
    elif np.issubdtype(audio_signal.dtype, np.floating):  # Check if the signal is of float type
        audio_signal /= np.max(np.abs(audio_signal))  # Normalize by the maximum absolute value in the signal

    if len(audio_signal.shape) == 2:  # Check if the signal is stereo
        audio_signal = np.mean(audio_signal, axis=1)  # Convert to mono by averaging the channels

    window = CreateHanningWindow(window_size)  # Create Hanning window of given length
    num_frames = 1 + (len(audio_signal) - window_size) // hop_size  # Compute the number of windows
    stft_matrix = np.zeros((window_size, num_frames), dtype=np.complex128)  # Create matrix to store STFT results

    for frame in range(num_frames):  # Loop over all windows
        start = frame * hop_size  # Compute start of the window
        end = start + window_size  # Compute end of the window
        frame_data = audio_signal[start:end] * window  # Multiply signal by Hanning window
        stft_matrix[:, frame] = FFT(frame_data)  # Compute FFT for the window and store result in matrix

    return stft_matrix, sample_rate  # Return STFT matrix and sample rate

# Function to compute the Inverse Short-Time Fourier Transform (ISTFT)
# Input: stft_matrix (numpy array) - the STFT matrix
#        sample_rate (int) - the sample rate
#        window_size (int) - the size of the window
#        hop_size (int) - the hop size
#        output_wav (str) - path to the output WAV file
# Output: None (writes the reconstructed signal to the output WAV file)
def ISTFT(stft_matrix, sample_rate, window_size, hop_size, output_wav):
    num_frames = stft_matrix.shape[1]  # Compute the number of windows
    expected_signal_length = window_size + hop_size * (num_frames - 1)  # Compute expected length of the reconstructed signal
    reconstructed_signal = np.zeros(expected_signal_length, dtype=np.float64)  # Create array to store reconstructed signal
    window = CreateHanningWindow(window_size)  # Create Hanning window of given length

    for frame in range(num_frames):  # Loop over all windows
        start = frame * hop_size  # Compute start of the window
        end = start + window_size  # Compute end of the window
        frame_data = IFFT(stft_matrix[:, frame])  # Compute IFFT for the window
        reconstructed_signal[start:end] += np.real(frame_data) * window  # Add reconstructed window to the reconstructed signal

    normalization = np.zeros_like(reconstructed_signal)  # Create array for normalization
    for frame in range(num_frames):  # Loop over all windows
        start = frame * hop_size  # Compute start of the window
        end = start + window_size  # Compute end of the window
        normalization[start:end] += window ** 2  # Compute normalization by squaring the window

    epsilon = 1e-8  # Small value to avoid division by zero
    reconstructed_signal /= (normalization + epsilon)  # Normalize the reconstructed signal

    reconstructed_signal = np.clip(reconstructed_signal, -1.0, 1.0)  # Clip the signal to the range [-1, 1]
    reconstructed_signal = (reconstructed_signal * 32767).astype(np.int16)  # Convert to int16 range

    wavfile.write(output_wav, sample_rate, reconstructed_signal)  # Write the reconstructed signal to the output WAV file

