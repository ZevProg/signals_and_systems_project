import numpy as np
import wave
import scipy.signal
import matplotlib.pyplot as plt


# Constants
CHUNK = 1500  # Number of samples per frame
MIN_PITCH = 50  # Minimum pitch frequency (Hz)
MAX_PITCH = 800  # Maximum pitch frequency (Hz)

# Create pitch estimation function
def pitch_estimation(frame, fs, min_pitch, max_pitch):
    # Constants
    MIN_PERIOD = fs // max_pitch
    MAX_PERIOD = fs // min_pitch
   
    # Remove mean
    frame -= np.mean(frame)
   
    # Apply Hanning window
    windowed_frame = frame * np.hanning(len(frame))
   
    # Autocorrelation
    corr = np.correlate(windowed_frame, windowed_frame, mode='full')
    corr = corr[len(corr) // 2:]
    
    # Thresholding the autocorrelation to remove low-magnitude peaks
    corr[corr < 0.1 * np.max(corr)] = 0
   
    # Find the first positive slope
    d = np.diff(corr)
    PositiveSlope=np.where(d > 0)
    if PositiveSlope[0].size == 0:
        start=0
    else:
        start = PositiveSlope[0][0]
   
    # Find the peak in the specified range
    peak = np.argmax(corr[start + MIN_PERIOD:start + MAX_PERIOD]) + start + MIN_PERIOD
   
    # Calculate pitch
    if peak >= MIN_PERIOD and peak <= MAX_PERIOD:
        # Refine peak using parabolic interpolation
        if peak > 0 and peak < len(corr) - 1:
            alpha = corr[peak - 1]
            beta = corr[peak]
            gamma = corr[peak + 1]
            p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma) if alpha - 2 * beta + gamma != 0 else 0
            pitch_period = peak + p
        else:
            pitch_period = peak
       
        pitch = fs / pitch_period
        if pitch > max_pitch or pitch < min_pitch:
            pitch = None
    else:
        pitch = None
    if(start==0):
        return None
    return pitch

# Convert wav file to numpy array and plot results for each frame.  used to show the pitch estimation over time
def process_wav_file_pitches(wf):
    # Open the WAV file
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
    for i in range(0, len(audio_data), CHUNK//2): #CHUNK//2 is used to overlap the frames
        frame = audio_data[i:i + CHUNK].astype(np.float32)
        if len(frame) == CHUNK:
            pitch = pitch_estimation(frame, framerate, MIN_PITCH, MAX_PITCH)
            pitches.append(pitch)
        
        # Optional: Apply median filter to smooth pitch estimates. only used for the plot
    #if len(pitches) > 0:
        #pitches = scipy.signal.medfilt(pitches, kernel_size=5)
        
        # Plot the detected pitches
    #plt.figure(figsize=(10, 6))
    #plt.plot(pitches, label='Detected Pitch')
    #plt.xlabel('Frame')
    #plt.ylabel('Pitch (Hz)')
    #plt.title('Pitch Estimation Over Time')
    #plt.legend()
    #plt.grid()
    #plt.show()

    return pitches #return the pitches array.



# EXAMPLE
#from PitchEstimation import process_wav_file_pitches #how to use the function
#file_path = 'C:/Users/Shira/שירה/אוניברסיטה/סמסטר ד/אותות ומערכות/עבודה חלק ב/cleaned_test1.wav' #path to the wav file
#open the wav file
#wf = wave.open(file_path, 'rb')
#pitches=process_wav_file_pitches(wf)
#print(pitches)
