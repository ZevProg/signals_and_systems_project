import numpy as np
import scipy.signal as signal
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
    analytic_signal = signal + 1j * np.imag(scipy.signal.hilbert(signal))
    ssb_signal = np.real(analytic_signal * np.exp(1j * 2 * np.pi * carrier_freq * t))
    return ssb_signal

# Demodulate the SSB signal
def ssb_demodulate(ssb_signal, carrier_freq, samplerate):
    t = np.arange(len(ssb_signal)) / samplerate
    demodulated_signal = ssb_signal * np.cos(2 * np.pi * carrier_freq * t)
    # Apply a low-pass filter to recover the original signal
    nyquist_rate = samplerate / 2.0
    cutoff = 4000  # Low-pass filter cutoff frequency
    b, a = signal.butter(5, cutoff / nyquist_rate)
    recovered_signal =  signal.lfilter(b, a, demodulated_signal)
    return recovered_signal

# Main process
input_filename = 'input.wav'
output_filename = 'output.wav'

# Load the input WAV file
data, samplerate = load_wav(input_filename)
carrier_freq = samplerate  # Carrier frequency in Hz

# Modulate the signal using SSB
ssb_signal = ssb_modulate(data, carrier_freq, samplerate)

# Demodulate the signal to recover the original data
recovered_signal = ssb_demodulate(ssb_signal, carrier_freq, samplerate)

# Save the recovered signal to an output WAV file
save_wav(output_filename, recovered_signal, samplerate)


# Plotting
t = np.arange(len(data)) / samplerate

plt.figure()
plt.subplot(3,1,1)
plt.plot(t, data)
plt.title('Original Baseband Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(3,1,2)
plt.plot(t, ssb_signal)
plt.title('SSB Modulated Signal (Upper Sideband)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(3,1,3)
plt.plot(t, recovered_signal)
plt.title('Demodulated Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()