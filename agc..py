import numpy as np
import wave
import struct

def read_wav(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        params = wav_file.getparams()
        frames = wav_file.readframes(params.nframes)
        wave_data = np.frombuffer(frames, dtype=np.int16)
        wave_data = wave_data.reshape(-1, params.nchannels)
        return wave_data, params

def write_wav(file_path, wave_data, params):
    with wave.open(file_path, 'wb') as wav_file:
        wav_file.setparams(params)
        frames = wave_data.tobytes()
        wav_file.writeframes(frames)

def calculate_rms(wave_data):
    return np.sqrt(np.mean(np.square(wave_data), axis=0))

def apply_agc(wave_data, target_rms):
    current_rms = calculate_rms(wave_data)
    gain = target_rms / (current_rms + 1e-6)  # Adding small value to prevent division by zero
    wave_data = wave_data * gain
    wave_data = np.clip(wave_data, -32768, 32767)  # Clipping to the range of int16
    return wave_data.astype(np.int16)

# File paths
input_path = '/Users/terner/Desktop/פרוייקט סיום קבוצתי אותות/input.wav'
output_path = '/Users/terner/Desktop/פרוייקט סיום קבוצתי אותות/output.wav'
# Parameters
target_rms = 10000  # Set the target RMS value

# Read, process, and write WAV file
wave_data, params = read_wav(input_path)
wave_data_agc = apply_agc(wave_data, target_rms)
write_wav(output_path, wave_data_agc, params)

print("AGC applied and output saved to", output_path)
