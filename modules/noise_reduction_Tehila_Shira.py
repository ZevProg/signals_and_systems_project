import wave
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import warnings

warnings.filterwarnings("ignore", category=wav.WavFileWarning)
#STFT part
def CreateHanningWindow(window_length):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(window_length) / (window_length - 1)))


def FFT(signal):
    N = len(signal)
    if N <= 1:
        return signal
    even = FFT(signal[0::2])
    odd = FFT(signal[1::2])
    T = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([even + T[:N // 2] * odd, even + T[N // 2:] * odd])


def IFFT(spectrum):
    N = len(spectrum)
    if N <= 1:
        return spectrum
    even = IFFT(spectrum[0::2])
    odd = IFFT(spectrum[1::2])
    T = np.exp(2j * np.pi * np.arange(N) / N)
    return (np.concatenate([even + T[:N // 2] * odd, even + T[N // 2:] * odd]) / 2)


def STFT(audio_signal, sample_rate=None, window_size=2048, hop_size=512):
    if isinstance(audio_signal, str):
        sample_rate, audio_signal = wav.read(audio_signal)
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

    return stft_matrix, sample_rate


def ISTFT(stft_matrix, sample_rate, window_size, hop_size, output_wav):
    num_frames = stft_matrix.shape[1]
    expected_signal_length = window_size + hop_size * (num_frames - 1)
    reconstructed_signal = np.zeros(expected_signal_length, dtype=np.float64)
    window = CreateHanningWindow(window_size)

    for frame in range(num_frames):
        start = frame * hop_size
        end = start + window_size
        frame_data = IFFT(stft_matrix[:, frame])
        reconstructed_signal[start:end] += np.real(frame_data) * window

    reconstructed_signal = reconstructed_signal * 32768
    reconstructed_signal = np.int16(reconstructed_signal / np.max(np.abs(reconstructed_signal)) * 32767)
    wav.write(output_wav, sample_rate, reconstructed_signal)


def check_file(input_file):
    return True #input_file.lower().endswith('.wav')


def NoiseReduction(input_file, output_file, speech_segments, frame_size, hop_size):
    if not check_file(input_file):
        print("This is not a WAV file")
        return None

    # Load input file
    print('Loading wav file:', input_file)
    sample_rate, waveform = wav.read(input_file)
    waveform = waveform.astype(np.float32) / 32768.0  # Normalize to -1 to 1 range


    # Ensure speech_segments length matches the number of STFT frames
    num_frames = 1 + (len(waveform) - frame_size) // hop_size
    speech_segments = np.array([int(s) for s in speech_segments.split(',')])
    if len(speech_segments) > num_frames:
        speech_segments = speech_segments[:num_frames]
    elif len(speech_segments) < num_frames:
        speech_segments = np.pad(speech_segments, (0, num_frames - len(speech_segments)), 'constant')

    # Perform STFT on the entire noisy signal
    stft_matrix, _ = STFT(waveform, sample_rate=sample_rate, window_size=frame_size, hop_size=hop_size)
    magnitude_spectrum = np.abs(stft_matrix)
    phase_spectrum = np.angle(stft_matrix)

    # Compute noise spectrum from non-speech segments
    noise_spectrum = np.zeros_like(magnitude_spectrum[:, 0])
    non_speech_frame_count = 0

    for i, is_speech in enumerate(speech_segments):
        if is_speech == 0:
            noise_spectrum += magnitude_spectrum[:, i]
            non_speech_frame_count += 1

    if non_speech_frame_count > 0:
        noise_spectrum /= non_speech_frame_count
    else:
        print("No non-speech frames detected, estimating noise from the first 0.25 seconds")
        noise_estimation_duration = 0.25
        noise_samples = int(noise_estimation_duration * sample_rate)
        noise_signal = waveform[:noise_samples]

        # Perform STFT on noise signal
        noise_stft_matrix, _ = STFT(noise_signal, sample_rate=sample_rate, window_size=frame_size, hop_size=hop_size)
        noise_magnitude_spectrum = np.abs(noise_stft_matrix)
        noise_spectrum = np.mean(noise_magnitude_spectrum, axis=1)

    mean_noise_spectrum = noise_spectrum

    # Noise reduction with oversubtraction and flooring
    alpha = 2  # Oversubtraction factor
    beta = 0.01  # Spectral floor
    cleaned_spectrum = np.maximum(magnitude_spectrum - alpha * mean_noise_spectrum.reshape((mean_noise_spectrum.shape[0], 1)), beta * magnitude_spectrum)

    # Reconstruct signal using inverse STFT
    cleaned_complex_spectrum = cleaned_spectrum * np.exp(1.0j * phase_spectrum)
    ISTFT(cleaned_complex_spectrum, sample_rate, window_size=frame_size, hop_size=hop_size, output_wav=output_file)
    print('Output wav file saved:', output_file)


