import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os

def read_audio(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    signal, samplerate = sf.read(file_path)
    if signal.ndim > 1:
        signal = signal[:, 0]  # Convert to mono if stereo
    return signal, samplerate

def write_audio(file_path, signal, samplerate):
    output_dir = os.path.dirname(file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sf.write(file_path, signal, samplerate)

def vad_agc(signal, vad_array, samplerate, target_level=0.5, attack_time=0.005, release_time=0.1, noise_floor=-60):
    # Convert time constants to samples
    attack_samples = int(attack_time * samplerate)
    release_samples = int(release_time * samplerate)

    # Initialize gain and envelope
    gain = 1.0
    envelope = np.zeros_like(signal)
    agc_signal = np.zeros_like(signal)
    noise_floor_linear = 10 ** (noise_floor / 20)

    if len(vad_array) != len(signal):
        raise ValueError("VAD array length must match the signal length")

    for i in range(len(signal)):
        # Calculate instantaneous level
        instant_level = abs(signal[i])

        # Update envelope only if VAD is active
        if vad_array[i] == 1:
            if instant_level > envelope[i-1]:
                envelope[i] = instant_level
            else:
                envelope[i] = envelope[i-1] * (1 - 1/release_samples)
        else:
            envelope[i] = envelope[i-1]

        # Calculate desired gain
        if envelope[i] > noise_floor_linear:
            desired_gain = target_level / (envelope[i] + 1e-6)  # Avoid division by zero
        else:
            desired_gain = gain  # Maintain current gain for noise floor

        # Smoothly approach desired gain
        if desired_gain < gain:
            gain = desired_gain + (gain - desired_gain) * np.exp(-1 / attack_samples)
        else:
            gain = desired_gain + (gain - desired_gain) * np.exp(-1 / release_samples)

        # Apply gain
        agc_signal[i] = signal[i] * gain

    return agc_signal, envelope, np.ones_like(signal) * gain

# Paths
input_audio_path = '/Users/terner/Desktop/פרוייקט סיום קבוצתי אותות/about_time.wav'
output_audio_path = '/Users/terner/Desktop/פרוייקט סיום קבוצתי אותות/output_vad_agc.wav'

# Read the input audio file
signal, samplerate = read_audio(input_audio_path)

# Generate a dummy VAD array (replace this with your actual VAD data)
# 1 indicates voice activity, 0 indicates no voice activity
vad_array = np.ones(len(signal))

# Apply VAD-aware AGC to the signal
agc_signal, envelope, gains = vad_agc(signal, vad_array, samplerate)

# Write the processed audio file
write_audio(output_audio_path, agc_signal, samplerate)



print(f"VAD-aware AGC applied and output saved to {output_audio_path}")
