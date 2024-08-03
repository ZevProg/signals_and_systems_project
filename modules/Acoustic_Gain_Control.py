import numpy as np

def vad_aware_agc_process(input_signal, binary_vector, sample_rate=44100):
    """
    Apply VAD-aware Automatic Gain Control (AGC) to the input WAV signal.

    Args:
    input_signal (np.array): Input audio signal (raw samples)
    binary_vector (np.array): VAD decision for each 10ms frame (1 for speech, 0 for non-speech)
    sample_rate (int): Sample rate of the input signal (default 44100 Hz)

    Returns:
    np.array: Processed audio signal after applying VAD-aware AGC
    """

    def compute_rms(signal, window_size):
        """
        Compute the Root Mean Square (RMS) of the signal using a sliding window.
        
        Args:
        signal (np.array): Input signal
        window_size (int): Size of the sliding window in samples
        
        Returns:
        np.array: RMS values
        """
        return np.sqrt(np.convolve(signal**2, np.ones(window_size)/window_size, mode='same'))

    # Constants
    frame_length_ms = 10  # Length of each frame in milliseconds
    frame_length_samples = int(frame_length_ms * sample_rate / 1000)  # Convert frame length to samples
    attack_time = 0.01  # Attack time in seconds
    release_time = 0.1  # Release time in seconds
    noise_floor = -60  # Noise floor in dB
    rms_window_ms = 50  # RMS window size in milliseconds

    # Convert time constants to sample counts
    attack_samples = int(attack_time * sample_rate)
    release_samples = int(release_time * sample_rate)
    rms_window_samples = int(rms_window_ms * sample_rate / 1000)

    # Compute target RMS
    target_rms = compute_rms(np.abs(input_signal), rms_window_samples)

    # Initialize arrays
    envelope = np.zeros_like(input_signal)  # Envelope of the signal
    agc_signal = np.zeros_like(input_signal)  # Output signal after AGC
    noise_floor_linear = 10 ** (noise_floor / 20)  # Convert noise floor from dB to linear scale

    # Check if VAD vector matches the number of frames
    num_frames = len(binary_vector)
    expected_frames = (len(input_signal) + frame_length_samples - 1) // frame_length_samples
    if num_frames != expected_frames:
        raise ValueError(f"VAD array length ({num_frames}) does not match the expected number of frames ({expected_frames})")

    gain = 1.0  # Initial gain

    # Process each sample
    for i in range(len(input_signal)):
        frame_index = i // frame_length_samples  # Determine which frame the current sample belongs to
        instant_level = abs(input_signal[i])  # Instantaneous level of the current sample

        # Update envelope based on VAD decision
        if binary_vector[frame_index] == 1:  # Speech frame
            if i == 0:
                envelope[i] = instant_level
            else:
                # Attack: quickly rise to new level
                envelope[i] = max(instant_level, envelope[i-1] * (1 - 1/release_samples))
        else:  # Non-speech frame
            if i == 0:
                envelope[i] = instant_level
            else:
                # Release: slowly fall to noise floor
                envelope[i] = envelope[i-1]

        # Compute desired gain
        if envelope[i] > noise_floor_linear:
            desired_gain = target_rms[i] / (envelope[i] + 1e-6)  # Avoid division by zero
        else:
            desired_gain = gain  # Maintain current gain if below noise floor

        # Apply attack/release to gain
        if desired_gain < gain:
            # Attack: quickly reduce gain
            gain = desired_gain + (gain - desired_gain) * np.exp(-1 / attack_samples)
        else:
            # Release: slowly increase gain
            gain = desired_gain + (gain - desired_gain) * np.exp(-1 / release_samples)

        # Apply gain and clip to prevent overflow
        agc_signal[i] = np.clip(input_signal[i] * gain, -1.0, 1.0)

    return agc_signal


print(f"VAD-aware AGC applied and output saved to {output_audio_path}")
