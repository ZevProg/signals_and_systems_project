import numpy as np
import wave
import matplotlib.pyplot as plt

# Constants
chunk = 1024  # Number of samples per frame
min_pitch = 50  # Minimum pitch frequency (Hz)
max_pitch = 800  # Maximum pitch frequency (Hz)

# The function get: frame from the audio, fs-The sampling frequency of the audio signal and the allowed range of the pitch.
# Create pitch estimation function
#The function plot: pitch of the frame of the audio.
def PitchEstimation(frame, fs, min_pitch, max_pitch):
    # Constants
    min_period = fs // max_pitch
    max_period = fs // min_pitch
    
    # Remove mean- remove the DC signal
    frame -= np.mean(frame)
    
    #Doubling the signal in the hamming window to soften the edges of the signal
    windowed_frame = frame * np.hamming(len(frame))
    
    # Autocorrelation ('full'=we take all the possible shift)
    corr = np.correlate(windowed_frame, windowed_frame, mode='full')
    #save only the positive shifts
    corr = corr[len(corr) // 2:]
    
    # Thresholding the autocorrelation to remove low-magnitude peaks 
    corr[corr < 0.1 * np.max(corr)] = 0
    
    # Find the first positive slope
    slope = np.diff(corr)
    start_increase = np.where(slope > 0)[0][0]
    
    # Find the peak in the specified range
    peak = np.argmax(corr[start_increase + min_period:start_increase + max_period]) + start_increase + min_period
    
    # Calculate pitch
    #Check that the peak is legal
    if peak >= min_period and peak <= max_period:
        # Refine peak using parabolic interpolation
        if peak > 0 and peak < len(corr) - 1:
            alpha = corr[peak - 1]
            beta = corr[peak]
            gamma = corr[peak + 1]
            p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
            pitch_period = peak + p
        else:
            pitch_period = peak
        
        pitch = fs / pitch_period
    else:
        pitch = 0
    
    return pitch

# Convert wav file to numpy array and plot results
#input: wav file, output:numpy array.
def ProcessWavFile(file_path):
    # Open the WAV file
    with wave.open(file_path, 'rb') as wf:
        num_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        num_frames = wf.getnframes()
        
        # Read the entire file
        audio_data = wf.readframes(num_frames)
        
        # Convert the byte data to numpy array
        audio_data = np.frombuffer(audio_data, dtype=np.int16)
        
        # If the audio is stereo, take only one channel
        if num_channels > 1:
            audio_data = audio_data[::num_channels]
        
        # Process the audio in chunks
        pitches = []
        for i in range(0, len(audio_data), chunk):
            frame = audio_data[i:i + chunk].astype(np.float32)
            if len(frame) == chunk:
                pitch = PitchEstimation(frame, framerate, min_pitch, max_pitch)
                pitches.append(pitch)
        
        # Optional: Apply median filter to smooth pitch estimates
        if len(pitches) > 0:
            pitches = scipy.signal.medfilt(pitches, kernel_size=5)
        
        # Plot the detected pitches
        plt.figure(figsize=(10, 6))
        plt.plot(pitches, label='Detected Pitch')
        plt.xlabel('Frame')
        plt.ylabel('Pitch (Hz)')
        plt.title('Pitch Estimation Over Time')
        plt.legend()
        plt.grid()
        plt.show()

# Process a WAV file
file_path = 'C:/Users/Shira/שירה/אוניברסיטה/סמסטר ד/אותות ומערכות/עבודה חלק ב/activity_unproductive.wav'
ProcessWavFile(file_path)
