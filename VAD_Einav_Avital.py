import wave
import numpy as np
import matplotlib.pyplot as plt


class VoiceActivityDetector:
    def __init__(self, filename, frame_duration=0.01, threshold=0.1, smoothness=0, remove_dc=True):
        self.filename = filename
        self.frame_duration = frame_duration
        self.threshold = threshold
        self.aggressiveness = smoothness
        self.look_back = self.get_look_back(smoothness)
        self.min_ones = 1  # This can be adjusted as needed
        self.remove_dc_flag = remove_dc
        self.audio_data, self.frame_rate = self.read_wav()
        self.original_audio_data = self.audio_data.copy()
        if self.remove_dc_flag:
            self.audio_data = self.remove_dc_component(self.audio_data)
        self.speech_segments = None
        self.energy = None
        self.smoothed_speech_segments = None

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
        times_energy = np.arange(len(self.energy)) * (len(self.audio_data) / len(self.energy) / self.frame_rate)
        plt.figure(figsize=(12, 4))
        plt.plot(times_energy, self.energy, 'b', alpha=0.3, label='Energy')
        plt.plot(times_energy, self.smoothed_speech_segments * np.max(self.energy), linewidth=2, label='Smoothed Speech Segments')
        plt.title('Voice Activity Detection')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Energy')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def print_speech_segments_info(self):
        # Print the binary sequence showing detected speech segments
        binary_sequence = ','.join(map(str, self.smoothed_speech_segments.astype(int)))
        print(binary_sequence)


def main():
    # audio_file= "C:\\temp\signal_system\\about_time.wav"
    # audio_file = "C:\\temp\\signal_system\\activity_unproductive.wav"
    # audio_file = "C:\\temp\signal_system\Heartbeat.wav"
    audio_file = "C:\\temp\\signal_system\\Counting.wav"
    smoothness = 3 # Choose between 0, 1, 2, or 3
    remove_dc = 1  # 1 to remove DC component, 0 to keep it
    vad = VoiceActivityDetector(audio_file, frame_duration=0.01, threshold=0.1, smoothness=smoothness, remove_dc=bool(remove_dc))
    vad.vad()
    vad.smooth_speech_segments()
    vad.plot_audio_data()
    vad.plot_vad_results()
    vad.print_speech_segments_info()


if __name__ == "__main__":
    main()


