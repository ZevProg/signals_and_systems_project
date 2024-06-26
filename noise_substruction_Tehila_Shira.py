import numpy as np
import scipy.io.wavfile as wav

def stft(x, fft_size=1024, hop_size=512):
    x = np.pad(x, int(fft_size // 2), mode='reflect')
    noverlap = fft_size - hop_size
    shape = (x.size - noverlap) // hop_size, fft_size
    strides = x.strides[0] * hop_size, x.strides[0]
    result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    result = np.fft.rfft(result, n=fft_size)
    return result

def istft(X, hop_size=512):
    fft_size = (X.shape[1] - 1) * 2
    x = np.zeros(X.shape[0] * hop_size + fft_size)
    for n, i in enumerate(range(0, len(x) - fft_size, hop_size)):
        x[i:i + fft_size] += np.fft.irfft(X[n])
    return x

# File paths
input_file = 'noisy_audio.wav'
output_file = 'cleaned_audio.wav'

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
noise_stft_matrix = stft(noise_signal, fft_size=frame_size, hop_size=hop_size)
noise_magnitude_spectrum = np.abs(noise_stft_matrix)
mean_noise_spectrum = np.mean(noise_magnitude_spectrum, axis=0)

# Perform STFT on the entire noisy signal
stft_matrix = stft(waveform, fft_size=frame_size, hop_size=hop_size)
magnitude_spectrum = np.abs(stft_matrix)
phase_spectrum = np.angle(stft_matrix)

# Noise reduction
cleaned_spectrum = magnitude_spectrum - mean_noise_spectrum.reshape((1, mean_noise_spectrum.shape[0]))
cleaned_spectrum = np.maximum(cleaned_spectrum, 0)  # Ensure no negative values

# Reconstruct signal using inverse STFT
cleaned_complex_spectrum = cleaned_spectrum * np.exp(1.0j * phase_spectrum)
output_waveform = istft(cleaned_complex_spectrum, hop_size=hop_size)

# Normalize and save as a wav file
output_waveform = output_waveform * 32768
output_waveform = np.int16(output_waveform / np.max(np.abs(output_waveform)) * 32767)  # Normalize to 16-bit range
wav.write(output_file, sample_rate, output_waveform)
print('Output wav file saved:', output_file)
