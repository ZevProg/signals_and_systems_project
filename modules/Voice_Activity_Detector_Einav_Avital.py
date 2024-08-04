import wave
import numpy as np
import matplotlib.pyplot as plt
import io
from io import BytesIO


class VoiceActivityDetector:
    def __init__(self, input_file, frame_duration, threshold, smoothness, remove_dc,
                 plot_graphs):
        """
        Initialize the VoiceActivityDetector with the given parameters.
        :param input_file: Input file object (BytesIO)
        :param frame_duration: Duration of each frame in seconds (0.01, 0.02, or 0.03)
        :param threshold: Energy threshold for speech detection (0 to 1)
        :param smoothness: Smoothness level for VAD (0, 1, 2, or 3)
        :param remove_dc: Boolean to remove DC component from the audio data
        :param plot_graphs: Boolean to plot graphs for debugging
        """
        # Parameters Section
        self.input_file = input_file
        self.frame_duration = frame_duration
        self.threshold = threshold
        self.aggressiveness = smoothness
        self.look_back = self.get_look_back(smoothness)
        self.min_ones = 1
        self.remove_dc_flag = remove_dc
        self.plot_graphs = plot_graphs

        # Validate Parameters
        self.validate_parameters()

        # Operations Section
        self.audio_data, self.frame_rate = self.read_wav(self.input_file)
        self.original_audio_data = self.audio_data.copy()
        if self.remove_dc_flag:
            self.audio_data = self.remove_dc_component(self.audio_data)
        self.speech_segments = None
        self.energy = None
        self.smoothed_speech_segments = None

    def validate_parameters(self):
        """
        Validate the parameters passed to the VoiceActivityDetector.
        :return: None
        """
        # Validate frame_duration
        if self.frame_duration not in [0.01, 0.02, 0.03]:
            raise ValueError("frame_duration must be 0.01, 0.02, or 0.03 seconds.")

        # Validate threshold
        if not (0 <= self.threshold <= 1):
            raise ValueError("threshold must be between 0 and 1.")

    @staticmethod
    def get_look_back(level):
        """
        Get the look-back window size based on the smoothness level.
        :param level: Smoothness level (0, 1, 2, or 3)
        :return: Look-back window size
        """
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

    def read_wav(self, input_file):
        """
        Read the WAV file and return the audio data and frame rate.
        :param input_file: Input file object
        :return: Tuple of audio data and frame rate
        """
        # Check if the file is a WAV file
        try:
            with wave.open(input_file, 'rb') as input_wav_file:
                # Get basic information
                n_channels = input_wav_file.getnchannels()
                sample_width = input_wav_file.getsampwidth()
                frame_rate = input_wav_file.getframerate()
                n_frames = input_wav_file.getnframes()

                # Read raw audio data
                raw_data = input_wav_file.readframes(n_frames)

        except Exception as X:
            raise ValueError("The provided file is not a WAV file.")

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
        :param audio_data: Numpy array of audio data
        :return: Numpy array of audio data without DC component
        """
        # Calculate the DC component (mean of the signal)
        dc_component = np.mean(audio_data)

        # Subtract the DC component from the signal to get the AC component
        ac_component = audio_data - dc_component

        return ac_component

    def vad(self):
        """
        Perform Voice Activity Detection on the audio data.
        :return: None
        """
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
        :return: None
        """
        smoothed_segments = self.speech_segments.copy()
        for i in range(len(self.speech_segments)):
            if smoothed_segments[i] == 0 and self.look_back > 0:
                # Check the previous `look_back` frames
                if i >= self.look_back and np.sum(self.speech_segments[i - self.look_back:i]) >= self.min_ones:
                    smoothed_segments[i] = 1
        self.smoothed_speech_segments = smoothed_segments

    def plot_audio_data(self):
        """
        Plot the audio data before and after DC removal.
        :return: None
        """
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
        """
        Plot the results of Voice Activity Detection.
        :return: None
        """
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
        """
        Get the binary sequence showing detected speech segments.
        :return: Binary sequence of detected speech segments
        """
        binary_sequence = ','.join(map(str, self.smoothed_speech_segments.astype(int)))
        return binary_sequence


def process_audio_file(input_file,frame_duration=0.01, threshold=0.1, smoothness=0, remove_dc=False, plot_graphs=False):
    """
    Process the audio file and perform Voice Activity Detection.
    :param input_file: Input file object
    :return: Binary sequence of detected speech segments
    """

    vad = VoiceActivityDetector(input_file, frame_duration, threshold, smoothness, remove_dc, plot_graphs)
    vad.vad()
    vad.smooth_speech_segments()
    vad.plot_audio_data()
    vad.plot_vad_results()
    return vad.get_speech_segments()


# def main():

    # input_file_path = "C:\\temp\signal_system\\Heartbeat.wav"
    # input_file_path ="C:\\temp\\signal_system\\Counting.wav"
    # input_file_path = "C:\\temp\signal_system\\about_time.wav"
    # input_file_path = "C:\\temp\\signal_system\\activity_unproductive.wav"
    # input_file_path = "C:\\temp\\vad_test\mp3.mp3"

    # with open(input_file_path, 'rb') as f:
        # output_wav = io.BytesIO(f.read())
        # output_wav.seek(0)

    # binary_vector = process_audio_file(output_wav)
    # print(binary_vector)
    # return binary_vector


# if __name__ == "__main__":
    # main()
