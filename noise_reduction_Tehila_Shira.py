import numpy as np
import scipy.io.wavfile as wav
import warnings

warnings.filterwarnings("ignore", category=wav.WavFileWarning)

# Function to create a Hanning window
def CreateHanningWindow(window_length):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(window_length) / (window_length - 1)))

# Function to compute the Fast Fourier Transform (FFT)
def FFT(signal):
    N = len(signal)
    if N <= 1:
        return signal
    even = FFT(signal[0::2])
    odd = FFT(signal[1::2])
    T = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([even + T[:N // 2] * odd, even + T[N // 2:] * odd])

# Function to compute the Inverse Fast Fourier Transform (IFFT)
def IFFT(spectrum):
    N = len(spectrum)
    if N <= 1:
        return spectrum
    even = IFFT(spectrum[0::2])
    odd = IFFT(spectrum[1::2])
    T = np.exp(2j * np.pi * np.arange(N) / N)
    return (np.concatenate([even + T[:N // 2] * odd, even + T[N // 2:] * odd]) / 2)

# Function to compute the Short-Time Fourier Transform (STFT)
def STFT(audio_signal, window_size=1024, hop_size=512):
    audio_signal = audio_signal.astype(np.float64)
    if np.issubdtype(audio_signal.dtype, np.integer):
        audio_signal /= np.iinfo(audio_signal.dtype).max
    elif np.issubdtype(audio_signal.dtype, np.floating):
        audio_signal /= np.max(np.abs(audio_signal))

    window = CreateHanningWindow(window_size)
    num_frames = 1 + (len(audio_signal) - window_size) // hop_size
    stft_matrix = np.zeros((window_size, num_frames), dtype=np.complex128)

    for frame in range(num_frames):
        start = frame * hop_size
        end = start + window_size
        frame_data = audio_signal[start:end] * window
        stft_matrix[:, frame] = FFT(frame_data)

    return stft_matrix

# Function to compute the Inverse Short-Time Fourier Transform (ISTFT)
def ISTFT(stft_matrix, hop_size=512):
    num_frames = stft_matrix.shape[1]
    window_size = stft_matrix.shape[0]
    expected_signal_length = window_size + hop_size * (num_frames - 1)
    reconstructed_signal = np.zeros(expected_signal_length, dtype=np.float64)
    window = CreateHanningWindow(window_size)

    for frame in range(num_frames):
        start = frame * hop_size
        end = start + window_size
        frame_data = IFFT(stft_matrix[:, frame])
        reconstructed_signal[start:end] += np.real(frame_data) * window

    return reconstructed_signal

def check_file(input_file):
    if input_file.lower().endswith('.wav'):
        return True
    else:
        return False

def NoiseReduction(input_file, output_file):
    if not check_file(input_file):
        print("This is not a WAV file")
        return None

    # Load input file
    print('Loading wav file:', input_file)
    sample_rate, waveform = wav.read(input_file)
    waveform = waveform.astype(np.float32) / 32768.0  # Normalize to -1 to 1 range

    # Parameters
    frame_size = 1024
    hop_size = 512

    # Use the first 0.25 seconds as noise estimation
    noise_duration = 0.25
    noise_samples = int(noise_duration * sample_rate)
    noise_signal = waveform[:noise_samples]

    # Perform STFT on noise signal using custom function
    noise_stft_matrix = STFT(noise_signal, window_size=frame_size, hop_size=hop_size)
    noise_magnitude_spectrum = np.abs(noise_stft_matrix)
    mean_noise_spectrum = np.mean(noise_magnitude_spectrum, axis=1)

    # Perform STFT on the entire noisy signal
    stft_matrix = STFT(waveform, window_size=frame_size, hop_size=hop_size)
    magnitude_spectrum = np.abs(stft_matrix)
    phase_spectrum = np.angle(stft_matrix)

    # Noise reduction
    cleaned_spectrum = magnitude_spectrum - mean_noise_spectrum.reshape((mean_noise_spectrum.shape[0], 1))
    cleaned_spectrum = np.maximum(cleaned_spectrum, 0)  # Ensure no negative values

    # Reconstruct signal using inverse STFT
    cleaned_complex_spectrum = cleaned_spectrum * np.exp(1.0j * phase_spectrum)
    output_waveform = ISTFT(cleaned_complex_spectrum, hop_size=hop_size)

    # Normalize and save as a wav file
    output_waveform = output_waveform * 32768
    output_waveform = np.int16(output_waveform / np.max(np.abs(output_waveform)) * 32767)  # Normalize to 16-bit range
    wav.write(output_file, sample_rate, output_waveform)
    print('Output wav file saved:', output_file)


#test1
input_file = 'test1_nr.wav'
output_file = 'cleaned_test1.wav'
#NoiseReduction(input_file,output_file) 


#test2
input_file = 'test2_nr.wav'
output_file = 'cleaned_test2.wav'
#NoiseReduction(input_file,output_file) 


#test3
input_file = 'test3_nr.wav'
output_file = 'cleaned_test3.wav'
NoiseReduction(input_file,output_file) 


#test4
input_file = 'test4_nr.wav'
output_file = 'cleaned_test4.wav'
NoiseReduction(input_file,output_file) 


#test5
input_file = 'noisy_audio.wav'
output_file = 'cleaned_noisy_audio.wav'  
NoiseReduction(input_file,output_file) 

#test6
input_file = 'mp3.MP3'
output_file = 'cleand_mp3.wav' 
NoiseReduction(input_file,output_file) 

