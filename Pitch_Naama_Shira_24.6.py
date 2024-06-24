import numpy as np
import wave
import scipy.signal as signal

# Constants
CHUNK = 1024  # Number of samples per frame
MIN_PITCH = 50  # Minimum pitch frequency (Hz)
MAX_PITCH = 500  # Maximum pitch frequency (Hz)

# Autocorrelation method for pitch detection
def autocorrelation_method(frame, fs, min_pitch, max_pitch):
    min_period = fs // max_pitch
    max_period = fs // min_pitch
    
    frame -= np.mean(frame)  # Remove mean
    corr = np.correlate(frame, frame, mode='full')
    corr = corr[len(corr) // 2:]  # Keep only the second half
    
    d = np.diff(corr)
    start = np.where(d > 0)[0][0]  # First positive slope
    peak = np.argmax(corr[start:]) + start
    
    if peak >= min_period and peak <= max_period:
        pitch_period = peak
        pitch = fs / pitch_period
    else:
        pitch = 0
    
    return pitch

# Function to process a WAV file
def process_wav_file(file_path):
    # Open the WAV file
    with wave.open(file_path, 'rb') as wf:
        num_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        num_frames = wf.getnframes()
        
        print(f"Channels: {num_channels}")
        print(f"Sample Width: {sampwidth}")
        print(f"Frame Rate: {framerate}")
        print(f"Number of Frames: {num_frames}")
        
        # Read the entire file
        audio_data = wf.readframes(num_frames)
        
        # Convert the byte data to numpy array
        audio_data = np.frombuffer(audio_data, dtype=np.int16)
        
        # If the audio is stereo, take only one channel
        if num_channels > 1:
            audio_data = audio_data[::num_channels]
        
        # Process the audio in chunks
        for i in range(0, len(audio_data), CHUNK):
            frame = audio_data[i:i + CHUNK].astype(np.float32)
            if len(frame) == CHUNK:
                pitch = autocorrelation_method(frame, framerate, MIN_PITCH, MAX_PITCH)
                print(f"Detected pitch: {pitch:.2f} Hz")


# Path to your WAV file
file_path = 'C:/Users/Shira/שירה/אוניברסיטה/סמסטר ד/אותות ומערכות/עבודה חלק ב/activity_unproductive.wav'
process_wav_file(file_path)
