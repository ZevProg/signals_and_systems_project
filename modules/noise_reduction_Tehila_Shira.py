import wave
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav


class VoiceActivityDetector:
    def __init__(self, filename, frame_duration=0.01, threshold=0.1, smoothness=0, remove_dc=True,
                 plot_graphs=False):
        # Parameters Section
        self.filename = filename
        self.frame_duration = frame_duration
        self.threshold = threshold
        self.aggressiveness = smoothness
        self.look_back = self.get_look_back(smoothness)
        self.min_ones = 1  # This can be adjusted a s needed
        self.remove_dc_flag = remove_dc
        self.plot_graphs = plot_graphs

        # Validate Parameters
        self.validate_parameters()

        # Check if the file is a WAV file
        if not self.is_wav_file(filename):
            raise ValueError("The provided file is not a WAV file.")

        # Operations Section
        self.audio_data, self.frame_rate = self.read_wav()
        self.original_audio_data = self.audio_data.copy()
        if self.remove_dc_flag:
            self.audio_data = self.remove_dc_component(self.audio_data)
        self.speech_segments = None
        self.energy = None
        self.smoothed_speech_segments = None

    def validate_parameters(self):
        # Validate frame_duration
        if self.frame_duration not in [0.01, 0.02, 0.03]:
            raise ValueError("frame_duration must be 0.01, 0.02, or 0.03 seconds.")

        # Validate threshold
        if not (0 <= self.threshold <= 1):
            raise ValueError("threshold must be between 0 and 1.")

    @staticmethod
    def is_wav_file(filename):
        return filename.lower().endswith('.wav')

    @staticmethod
    def get_look_back(level):
        if level == 0:
            return 0
        elif level == 1:
            return 4
        elif level == 2:
            return 6
        elif level == 3:
            return 8
        else:
            raise ValueError("Invalid aggressiveness level. Choose between 0, 1, 2, or 3.")

    def read_wav(self):
        with wave.open(self.filename, 'rb') as wav_file:
            # Get basic information
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()

            # Read raw audio data
            raw_data = wav_file.readframes(n_frames)

        # Convert raw data to numpy array
        if sample_width == 1:
            dtype = np.int8
        elif sample_width == 2:
            dtype = np.int16
        else:
            raise ValueError("Unsupported sample width")

        audio_data = np.frombuffer(raw_data, dtype=dtype)

        # If stereo, take the mean of both channels
        if n_channels == 2:
            audio_data = audio_data.reshape(-1, 2).mean(axis=1)

        # Convert to int32 for further processing
        audio_data = audio_data.astype(np.int32)

        return audio_data, frame_rate

    def remove_dc_component(self, audio_data):
        """
        Removes the DC component from the audio data.
        """
        # Calculate the DC component (mean of the signal)
        dc_component = np.mean(audio_data)

        # Subtract the DC component from the signal to get the AC component
        ac_component = audio_data - dc_component

        return ac_component

    def vad(self):
        # Calculate frame size
        frame_size = int(self.frame_rate * self.frame_duration)

        # Pad the audio data to ensure all frames are full
        remainder_size = (len(self.audio_data) % frame_size)
        if remainder_size > 0:
            pad_size = frame_size - remainder_size
            self.audio_data = np.pad(self.audio_data, (0, pad_size), 'constant')

        # Reshape audio data into frames
        frames = self.audio_data.reshape(-1, frame_size)

        # Convert frames to float32 before squaring to avoid overflow
        frames = frames.astype(np.float32)

        # Calculate energy for each frame
        self.energy = np.sum(frames ** 2, axis=1) / frame_size

        # Normalize the energy values
        max_energy = np.max(self.energy)
        self.energy = self.energy / max_energy if max_energy != 0 else self.energy

        # Apply threshold to get speech segments
        self.speech_segments = self.energy > self.threshold

    def smooth_speech_segments(self):
        """
        Smooths the speech segments based on a look-back window.
        """
        smoothed_segments = self.speech_segments.copy()
        for i in range(len(self.speech_segments)):
            if smoothed_segments[i] == 0 and self.look_back > 0:
                # Check the previous `look_back` frames
                if i >= self.look_back and np.sum(self.speech_segments[i - self.look_back:i]) >= self.min_ones:
                    smoothed_segments[i] = 1
        self.smoothed_speech_segments = smoothed_segments

    def plot_audio_data(self):
        if not self.plot_graphs:
            return
        times_audio = np.arange(len(self.audio_data)) / self.frame_rate
        times_original_audio = np.arange(len(self.original_audio_data)) / self.frame_rate
        plt.figure(figsize=(12, 4))
        plt.plot(times_original_audio, self.original_audio_data, 'b', alpha=0.5, label='Original Audio')
        if self.remove_dc_flag:
            plt.plot(times_audio, self.audio_data, 'g', alpha=0.5, label='Audio After DC Removal')
        plt.title('Audio Data Before and After DC Removal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_vad_results(self):
        if not self.plot_graphs:
            return
        times_energy = np.arange(len(self.energy)) * (len(self.audio_data) / len(self.energy) / self.frame_rate)
        plt.figure(figsize=(12, 4))
        plt.plot(times_energy, self.energy, 'b', alpha=0.3, label='Energy')
        plt.plot(times_energy, self.smoothed_speech_segments * np.max(self.energy), linewidth=2,
                 label='Smoothed Speech Segments')
        plt.axhline(y=self.threshold, color='b', linestyle='--', label='Threshold')
        plt.title('Voice Activity Detection')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Energy')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def get_speech_segments(self):
        binary_sequence = ','.join(map(str, self.smoothed_speech_segments.astype(int)))
        return binary_sequence


def process_audio_file(filename):
    # Default parameters
    frame_duration = 0.01  # Choose between 0.01, 0.02, or 0.03
    threshold = 0.1  # Must be between 0 and 1
    smoothness = 3  # Choose between 0, 1, 2, or 3
    remove_dc = True  # True to remove DC component, False to keep it
    plot_graphs = False  # Set to True to plot graphs, False to disable plotting

    vad = VoiceActivityDetector(filename, frame_duration, threshold, smoothness, remove_dc, plot_graphs)
    vad.vad()
    vad.smooth_speech_segments()
    vad.plot_audio_data()
    vad.plot_vad_results()
    return vad.get_speech_segments()
 


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
    return input_file.lower().endswith('.wav')

def NoiseReduction(input_file, output_file, speech_segments):
    if not check_file(input_file):
        print("This is not a WAV file")
        return None

    # Load input file
    print('Loading wav file:', input_file)
    sample_rate, waveform = wav.read(input_file)
    waveform = waveform.astype(np.float32) / 32768.0  # Normalize to -1 to 1 range

    # Parameters
    frame_size = 2048
    hop_size = 512

    # Ensure speech_segments length matches the number of STFT frames
    num_frames = 1 + (len(waveform) - frame_size) // hop_size
    if len(speech_segments) > num_frames:
        speech_segments = speech_segments[:num_frames]
    elif len(speech_segments) < num_frames:
        speech_segments = np.pad(speech_segments, (0, num_frames - len(speech_segments)), 'constant')

    # Perform STFT on the entire noisy signal
    stft_matrix = STFT(waveform, window_size=frame_size, hop_size=hop_size)
    magnitude_spectrum = np.abs(stft_matrix)
    phase_spectrum = np.angle(stft_matrix)

    # Compute noise spectrum from non-speech segments
    noise_spectrum = np.zeros_like(magnitude_spectrum[:, 0])
    non_speech_frame_count = 0

    for i, is_speech in enumerate(speech_segments):
        if is_speech=='0':
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
        noise_stft_matrix = STFT(noise_signal, window_size=frame_size, hop_size=hop_size)
        noise_magnitude_spectrum = np.abs(noise_stft_matrix)
        noise_spectrum = np.mean(noise_magnitude_spectrum, axis=1)

    mean_noise_spectrum = noise_spectrum

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


def main():
    # Parameters
    # audio_file= "C:\\temp\signal_system\\about_time.wav"
    # audio_file = "C:\\temp\\signal_system\\activity_unproductive.wav"
    # audio_file = "C:\\temp\signal_system\Heartbeat.wav"
    #test1
    input_file = 'test1_nr.wav'
    output_file = 'cleaned_test1.wav'
    binary_vector = process_audio_file(input_file)
    print (binary_vector)
    NoiseReduction(input_file,output_file,binary_vector) 


    #test2
    input_file = 'test2_nr.wav'
    output_file = 'cleaned_test2.wav'
    binary_vector = process_audio_file(input_file)
    print (binary_vector)
    NoiseReduction(input_file,output_file,binary_vector) 
 


    #test3
    input_file = 'test3_nr.wav'
    output_file = 'cleaned_test3.wav'
    binary_vector = process_audio_file(input_file)
    print (binary_vector)
    NoiseReduction(input_file,output_file,binary_vector) 


    #test4
    input_file = 'test4_nr.wav'
    output_file = 'cleaned_test4.wav'
    binary_vector = process_audio_file(input_file)
    print (binary_vector)
    NoiseReduction(input_file,output_file,binary_vector) 
 


    #test5
    input_file = 'noisy_audio.wav'
    output_file = 'cleaned_noisy_audio.wav'
    binary_vector = process_audio_file(input_file)
    print (binary_vector)
    NoiseReduction(input_file,output_file,binary_vector) 
  

    #test6
    input_file = 'mp3.MP3'
    output_file = 'cleand_mp3.wav' 
    binary_vector = process_audio_file(input_file)
    print (binary_vector)
    NoiseReduction(input_file,output_file,binary_vector) 



if __name__ == "__main__":
    main()

# for integration - constant parameters:
# from VAD_for_chain import process_audio_file
# audio_file = "C:\\temp\\signal_system\\Counting.wav"
# binary_vector = process_audio_file(audio_file)
# print(binary_vector)
