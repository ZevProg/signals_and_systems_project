import numpy as np
import wave
import io

def vad_aware_agc_process(input_signal_bytes, binary_vector):
    """
    Apply VAD-aware Automatic Gain Control (AGC) to the input audio signal.

    Args:
    input_signal_bytes (bytes): Input audio data as bytes
    binary_vector (np.array): VAD decision for each frame (1 for speech, 0 for non-speech)

    Returns:
    np.array: Processed audio signal after applying VAD-aware AGC
    """
    def extract_audio_data(audio_bytes):
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
            channels = wav_file.getnchannels()
            bit_depth = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            audio_data = wav_file.readframes(frame_count)

        if bit_depth == 1:
            data_type = np.int8
        elif bit_depth == 2:
            data_type = np.int16
        else:
            raise ValueError("Unsupported bit depth in the audio file.")

        audio_array = np.frombuffer(audio_data, dtype=data_type)

        if channels == 2:
            audio_array = audio_array.reshape(-1, 2).mean(axis=1)

        return audio_array.astype(np.float32) / np.iinfo(data_type).max, sample_rate

    def compute_rms(signal, window_size):
        return np.sqrt(np.convolve(signal**2, np.ones(window_size)/window_size, mode='same'))

    def calculate_parameters(input_signal, sample_rate):
        signal_duration = len(input_signal) / sample_rate
        signal_energy = np.sum(input_signal**2)
        
        attack_time = max(0.001, min(0.02, 0.01 * (signal_duration / 10)))
        release_time = max(0.05, min(0.2, 0.1 * (signal_duration / 10)))
        noise_floor = max(-80, min(-40, -60 + 10 * np.log10(signal_energy)))
        rms_window_ms = max(20, min(100, 50 * (signal_duration / 10)))
        
        return attack_time, release_time, noise_floor, rms_window_ms

    # Extract audio data and sample rate
    input_signal, sample_rate = extract_audio_data(input_signal_bytes)

    # Calculate adaptive parameters
    attack_time, release_time, noise_floor, rms_window_ms = calculate_parameters(input_signal, sample_rate)

    # Constants
    frame_duration = 0.01  # 10ms frame duration

    # Convert time constants to sample counts
    frame_length_samples = int(frame_duration * sample_rate)
    attack_samples = int(attack_time * sample_rate)
    release_samples = int(release_time * sample_rate)
    rms_window_samples = int(rms_window_ms * sample_rate / 1000)

    # Calculate the number of frames based on the input signal length
    num_samples = len(input_signal)
    num_frames = (num_samples + frame_length_samples - 1) // frame_length_samples

    # Ensure the VAD binary vector matches the number of frames
    if len(binary_vector) < num_frames:
        # Pad the binary vector if it's too short
        binary_vector = np.pad(binary_vector, (0, num_frames - len(binary_vector)), mode='constant', constant_values=0)
    elif len(binary_vector) > num_frames:
        # Truncate the binary vector if it's too long
        binary_vector = binary_vector[:num_frames]

    # Now binary_vector length should match num_frames
    assert len(binary_vector) == num_frames, f"VAD array length ({len(binary_vector)}) does not match the number of frames ({num_frames})"

    # Compute target RMS
    target_rms = compute_rms(np.abs(input_signal), rms_window_samples)

    # Initialize arrays
    envelope = np.zeros_like(input_signal)
    agc_signal = np.zeros_like(input_signal)
    noise_floor_linear = 10 ** (noise_floor / 20)

    gain = 1.0

    # Process each sample
    for i in range(len(input_signal)):
        frame_index = i // frame_length_samples
        instant_level = abs(input_signal[i])

        # Update envelope based on VAD decision
        if binary_vector[frame_index] == 1:
            if i == 0:
                envelope[i] = instant_level
            else:
                envelope[i] = max(instant_level, envelope[i-1] * (1 - 1/release_samples))
        else:
            if i == 0:
                envelope[i] = instant_level
            else:
                envelope[i] = envelope[i-1]

        # Compute desired gain
        if envelope[i] > noise_floor_linear:
            desired_gain = target_rms[i] / (envelope[i] + 1e-6)
        else:
            desired_gain = gain

        # Apply attack/release to gain
        if desired_gain < gain:
            gain = desired_gain + (gain - desired_gain) * np.exp(-1 / attack_samples)
        else:
            gain = desired_gain + (gain - desired_gain) * np.exp(-1 / release_samples)

        # Apply gain and clip to prevent overflow
        agc_signal[i] = np.clip(input_signal[i] * gain, -1.0, 1.0)

    return agc_signal

# Example usage:
# processed_signal = vad_aware_agc_process(audio_bytes, binary_vector)
