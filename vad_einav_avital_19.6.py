import os
import wave
import webrtcvad
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
from pydub import AudioSegment
import matplotlib.pyplot as plt


def voice_detector(file_path, vad_mode=3, frame_duration_ms=10):
    """
    Process a WAV file to convert it to a binary signal using Voice Activity Detection (VAD).

    Parameters:
    file_path (str): Path to the WAV file to be processed.
    vad_mode (int): VAD aggressiveness mode (0-3), default is 3.
    frame_duration_ms (int): Frame duration in milliseconds (10, 20, 30), default is 10.

    Returns:
    None
    """
    audio = AudioSegment.from_wav(file_path)

    # Convert to mono if not already
    if audio.channels != 1:
        audio = audio.set_channels(1)

    # Ensure the audio is 16-bit PCM
    if audio.sample_width != 2:
        audio = audio.set_sample_width(2)

    # Export the processed audio to a new WAV file
    processed_file_path = file_path.replace(".wav", "_processed.wav")
    audio.export(processed_file_path, format="wav")

    # Open the processed WAV file
    wav_file = wave.open(processed_file_path, "r")

    # Create a WebRTCVAD object
    vad = webrtcvad.Vad()
    vad.set_mode(vad_mode)  # Set the VAD mode (0-3)

    # Initialize variables
    binary_signal = []

    # Read and process frames
    frame_size = int(wav_file.getframerate() * frame_duration_ms / 1000)
    while True:
        data = wav_file.readframes(frame_size)
        if len(data) < frame_size * wav_file.getsampwidth():
            break

        # Check for voice activity
        is_speech = vad.is_speech(data, wav_file.getframerate())

        # Append 1 (talking) or 0 (silence/background noise) to the binary signal
        binary_signal.append(1 if is_speech else 0)

    # Close the WAV file
    wav_file.close()

    # Count the number of ones in the binary signal
    number_of_ones = binary_signal.count(1)

    # Print the binary signal and the count of ones
    print("Binary Signal:", binary_signal)
    print("Number of ones (speech frames):", number_of_ones)

    # Plot the binary signal
    plt.figure(figsize=(10, 4))
    plt.plot(binary_signal, drawstyle='steps-pre')
    plt.xlabel('Frame Number')
    plt.ylabel('Speech (1) / Silence (0)')
    plt.title('Voice Activity Detection')
    plt.grid(True)
    plt.show()


def downsample_wav(src, dst, outrate):
    """
    Downsamples a WAV audio file from a source path 'src' to a destination path 'dst'.

    Parameters:
    src (str): Source path of the WAV file to be downsampled.
    dst (str): Destination path where the downsampled WAV file will be saved.
    outrate (int): Output sampling rate of the destination WAV file.

    Returns:
    bool: True if the downsampling process was successful, False otherwise.
    """
    # Read the WAV file
    inrate, audio_data = wavfile.read(src)

    # Ensure the audio data is in mono
    if audio_data.ndim == 2:
        audio_data = audio_data.mean(axis=1)

    # Resample the audio data
    number_of_samples = round(len(audio_data) * float(outrate) / inrate)
    downsampled_data = signal.resample(audio_data, number_of_samples).astype(audio_data.dtype)

    # Write the downsampled audio data to destination WAV file
    with wave.open(dst, 'wb') as s_write:
        s_write.setnchannels(1)
        s_write.setsampwidth(audio_data.dtype.itemsize)
        s_write.setframerate(outrate)
        s_write.writeframes(downsampled_data.tobytes())

    return True


def determine_target_rate(current_rate):
    """
    Determine the target sampling rate based on the current rate.

    Parameters:
    current_rate (int): Current sampling rate of the audio file.

    Returns:
    int: Target sampling rate.
    """
    if current_rate > 48000:
        return 48000
    elif 32000 < current_rate <= 48000:
        return 32000
    elif 16000 < current_rate <= 32000:
        return 16000
    elif 8000 < current_rate <= 16000:
        return 8000
    else:
        raise ValueError(f"Unsupported sampling rate: {current_rate}. Must be above 8000 Hz.")


def main(file_path, vad_mode=3, frame_duration_ms=10):
    """
    Main function to process a WAV file.

    Parameters:
    file_path (str): Path to the WAV file to be processed.
    vad_mode (int): VAD aggressiveness mode (0-3), default is 3.
    frame_duration_ms (int): Frame duration in milliseconds (10, 20, 30), default is 10.

    Returns:
    None
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Get the current sampling rate of the file
    current_rate = AudioSegment.from_wav(file_path).frame_rate

    try:
        # Determine the target sampling rate
        target_rate = determine_target_rate(current_rate)
    except ValueError as e:
        print(e)
        return

    # Extract file name and extension
    file_dir, file_name = os.path.split(file_path)
    file_base, file_ext = os.path.splitext(file_name)

    # If the current rate is not the target rate, downsample the file
    if current_rate != target_rate:
        new_file_path = os.path.join(file_dir, f"{file_base}_downsampled{file_ext}")
        downsample_wav(file_path, new_file_path, target_rate)
        file_path = new_file_path

    # Process the file to convert it to a binary signal
    voice_detector(file_path, vad_mode, frame_duration_ms)


if __name__ == "__main__":
    """
    please note that currently the sample files are written in the path on our computer, 
    so you will have to download it to yours and then put a new path :)
    Also you will need to download all the libraries above (yes, including pip install)

    real rate -> target rate
    """

    # 44100 -> 32000
    file_path = "C:\\temp\signal_system\Heartbeat.wav"

    # 22050 -> 16000
    # file_path = "C:\\temp\\signal_system\\activity_unproductive.wav"

    # 16000
    # file_path = "C:\\temp\signal_system\Counting.wav"

    # 11025 -> 8000
    # file_path = "C:\\temp\\signal_system\\about_time.wav"

    # vad_mode - 0/1/2/3
    # frame_duration_ms - 10/20/30
    main(file_path, vad_mode=3, frame_duration_ms=10)
