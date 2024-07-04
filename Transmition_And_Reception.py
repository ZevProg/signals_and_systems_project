from random import sample
import numpy as np
import scipy.signal as signal
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import scipy

# Load the WAV file
def load_wav(filename):
    data, samplerate = sf.read(filename)
    if data.ndim > 1:  # Handle stereo by taking only one channel
        data = data[:, 0]
    return data, samplerate

# Save the WAV file
def save_wav(filename, data, samplerate):
    sf.write(filename, data, samplerate)

# Generate a SSB signal
def ssb_modulate(signal, carrier_freq, samplerate):
    t = np.arange(len(signal)) / samplerate
    y1 = signal * np.cos(2 * np.pi * carrier_freq * t)
    y2 = np.imag(scipy.signal.hilbert(signal)) * np.sin(2 * np.pi * carrier_freq * t)
    ssb_signal = y1 - y2
    return ssb_signal

# Demodulate the SSB signal
def ssb_demodulate(ssb_signal, carrier_freq, samplerate):
    t = np.arange(len(ssb_signal)) / samplerate
    demodulated_signal = ssb_signal * np.cos(2 * np.pi * carrier_freq * t)
    # Apply a low-pass filter to recover the original signal
    nyquist_rate = samplerate / 2.0
    cutoff = 4000  # Low-pass filter cutoff frequency
    b, a = signal.butter(5, cutoff / nyquist_rate)
    recovered_signal = signal.lfilter(b, a, demodulated_signal)
    return recovered_signal

# Real-time recording and processing callback
def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(status, flush=True)
    
    ssb_signal = ssb_modulate(indata[:, 0], carrier_freq, samplerate)
    recovered_signal = ssb_demodulate(ssb_signal, carrier_freq, samplerate)
    
    outdata[:, 0] = recovered_signal
    outdata[:, 1] = recovered_signal

# Main function
def main(mode='file', filename=None, duration=5, carrier_freq=10000):
    global samplerate
    if mode == 'file' and filename:
        # Load the input WAV file
        data, samplerate = load_wav(filename)

        # Modulate the signal using SSB
        ssb_signal = ssb_modulate(data, carrier_freq, samplerate)

        # Demodulate the signal to recover the original data
        recovered_signal = ssb_demodulate(ssb_signal, carrier_freq, samplerate)

        # Save the recovered signal to an output WAV file
        output_filename = 'output_' + filename
        save_wav(output_filename, recovered_signal, samplerate)

        # Plotting
        t = np.arange(len(data)) / samplerate

        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(t, data)
        plt.title('Original Baseband Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.subplot(3, 1, 2)
        plt.plot(t, ssb_signal)
        plt.title('SSB Modulated Signal (Upper Sideband)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.subplot(3, 1, 3)
        plt.plot(t, recovered_signal)
        plt.title('Demodulated Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.tight_layout()
        plt.show()

    elif mode == 'live':
        # Start the stream for real-time recording and processing
        with sd.Stream(samplerate=samplerate, channels=2, callback=audio_callback):
            print("Recording and processing in real-time. Press Ctrl+C to stop.")
            sd.sleep(int(duration * 1000))

        # Record a bit of data for plotting purposes
        print("Recording for plotting purposes...")
        recorded_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
        sd.wait()

        t = np.arange(len(recorded_data)) / samplerate
        save_wav('recording.wav',recorded_data,samplerate)
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(t, recorded_data)
        plt.title('Original Baseband Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        ssb_signal = ssb_modulate(recorded_data[:, 0], carrier_freq, samplerate)
        plt.subplot(3, 1, 2)
        plt.plot(t, ssb_signal)
        plt.title('SSB Modulated Signal (Upper Sideband)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        recovered_signal = ssb_demodulate(ssb_signal, carrier_freq, samplerate)
        save_wav('recording_modulated.wav',recovered_signal,samplerate)
        plt.subplot(3, 1, 3)
        plt.plot(t, recovered_signal)
        plt.title('Demodulated Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.tight_layout()
        plt.show()
    else:
        print("Invalid mode or filename not provided for file mode.")

# Set the sample rate and carrier frequency
samplerate = 44100
carrier_freq = 10000

main(mode='file',filename='file_name.wav')
main(mode='live', duration=5, carrier_freq=carrier_freq)
