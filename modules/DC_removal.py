import wave
import numpy as np
from scipy.io import wavfile
import io

def DC_Removal_filter(input_file, output_file, cutoff_frequency=100, numtaps=4400):
    # Check if the input file is a WAV file
    if not isinstance(input_file, (io.BytesIO, wave.Wave_read)):
        return "Error: Input must be a WAV file object."

    def sinc_high_pass_filter(cutoff, fs, numtaps):
        t = np.arange(numtaps) - (numtaps - 1) / 2
        sinc_func = np.sinc(2 * cutoff * t / fs)
        window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (numtaps - 1)) for n in range(numtaps)])
        sinc_func *= window
        sinc_func /= np.sum(sinc_func)
        delta = np.zeros(numtaps)
        delta[(numtaps - 1) // 2] = 1
        return delta - sinc_func

    # Read the WAV file
    with wave.open(input_file, 'rb') as wav_file:
        fs = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        n_frames = wav_file.getnframes()
        
        # Read all frames
        frames = wav_file.readframes(n_frames)
        
    # Convert frames to numpy array
    signal = np.frombuffer(frames, dtype=np.int16)
    
    # Convert to float32 for processing
    signal = signal.astype(np.float32)
    
    # If stereo, convert to mono by averaging channels
    if n_channels == 2:
        signal = signal.reshape(-1, 2).mean(axis=1)
    
    # Normalize the signal
    signal = signal / np.max(np.abs(signal))
    
    # Get the high-pass filter coefficients
    high_pass_filter = sinc_high_pass_filter(cutoff_frequency, fs, numtaps)
    
    # Initialize output signal
    output_signal = np.zeros_like(signal)
    
    # Buffer to store previous input values (length of filter coefficients)
    buffer = np.zeros(numtaps)
    
    # Apply the filter sample-by-sample
    for n in range(len(signal)):
        # Update the buffer with the current input sample
        buffer = np.roll(buffer, 1)
        buffer[0] = signal[n]
        # Compute the output sample by convolving the filter coefficients with the buffer
        output_signal[n] = np.dot(high_pass_filter, buffer)
    
    # Convert back to int16
    output_signal = (output_signal * 32767).astype(np.int16)
    
    # Write the filtered signal to a new WAV file
    with wave.open(output_file, 'wb') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(fs)
        wav_file.writeframes(output_signal.tobytes())
    
    return output_file
