from random import sample
import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib.pyplot as plt
import scipy
import sounddevice as sd
import threading
import os

def SSB(mode='file', file=None, carrier_freq=10000, samplerate=44100):

    # Load the WAV file
    def load_wav(file):
        data, samplerate = sf.read(file)
        if len(data) == 0:
            raise ValueError("The WAV file is empty.")
        if data.ndim > 1:  # Handle stereo by taking only one channel
            data = data[:, 0]  # Use only the first channel for stereo files
        return data, samplerate

    # Save the WAV file
    def save_wav(filename, data, samplerate):
        sf.write(filename, data, samplerate)

    # Generate a SSB signal
    def ssb_modulate(signal, carrier_freq, samplerate):
        t = np.arange(len(signal)) / samplerate
        signal_mul_cos = signal * np.cos(2 * np.pi * carrier_freq * t)  # Multiply with cosine
        signal_hilbert = scipy.signal.hilbert(signal)  # Apply Hilbert transform
        signal_mul_sin = np.imag(signal_hilbert) * np.sin(2 * np.pi * carrier_freq * t)  # Multiply with sine
        ssb_signal = signal_mul_cos - signal_mul_sin  # SSB signal (Upper Sideband)
        return ssb_signal

    # Low-pass filter using FFT
    def low_pass_filter(signal, cutoff_freq, samplerate):
        fft_signal = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1 / samplerate)
        filtered_fft_signal = fft_signal.copy()
        filtered_fft_signal[np.abs(freqs) > cutoff_freq] = 0  # Zero out frequencies above cutoff
        filtered_signal = 2 * np.fft.ifft(filtered_fft_signal)
        return np.real(filtered_signal)

    # Demodulate the SSB signal
    def ssb_demodulate(ssb_signal, carrier_freq, samplerate):
        t = np.arange(len(ssb_signal)) / samplerate
        demodulated_signal = ssb_signal * np.cos(2 * np.pi * carrier_freq * t)  # Multiply with cosine
        cutoff_freq = 4000  # Low-pass filter cutoff frequency
        recovered_signal = low_pass_filter(demodulated_signal, cutoff_freq, samplerate)  # Apply low-pass filter
        return recovered_signal

    # Real-time recording and processing callback
    def audio_callback(indata, outdata, frames, time, status):
        if status:
            print(status, flush=True)  # Print status if there's an error

        ssb_signal = ssb_modulate(indata[:, 0], carrier_freq, samplerate)  # Modulate input signal
        recovered_signal = ssb_demodulate(ssb_signal, carrier_freq, samplerate)  # Demodulate signal

        if outdata is not None:
            outdata[:, 0] = recovered_signal  # Output recovered signal to both channels
            outdata[:, 1] = recovered_signal

    if mode == 'file' and file:
        try:
            # Load the input WAV file
            data, samplerate = load_wav(file)
            carrier_freq = samplerate / 2  # Set carrier frequency to half the sample rate

            # Modulate the signal using SSB
            ssb_signal = ssb_modulate(data, carrier_freq, samplerate)

            # Demodulate the signal to recover the original data
            recovered_signal = 0.5 * ssb_demodulate(ssb_signal, carrier_freq, samplerate)

            # Save the recovered signal to an output WAV file
            output_filename = 'output_file.wav'
            save_wav(output_filename, recovered_signal, samplerate)

            # Plotting the signals
            t = np.arange(len(data)) / samplerate

            plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(np.fft.fftfreq(len(t)), np.fft.fft(data))
            plt.title('Original Baseband Signal')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')

            plt.subplot(3, 1, 2)
            plt.plot(np.fft.fftfreq(len(t)), np.fft.fft(ssb_signal))
            plt.title('SSB Modulated Signal (Upper Sideband)')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')

            plt.subplot(3, 1, 3)
            plt.plot(np.fft.fftfreq(len(t)), np.fft.fft(recovered_signal))
            plt.title('Demodulated Signal')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')

            plt.tight_layout()
            plt.show()

            return output_filename
        
        except FileNotFoundError as e:
            print(e)
        except ValueError as e:
            print(e)

    # 'live' option is in beta
    # elif mode == 'live':
    #     global recorded_data
    #     recorded_data = []
    #
    #     # Thread for stopping the recording on Enter key press
    #     def wait_for_enter():
    #         input("Press Enter to stop recording...\n")
    #         sd.stop()
    #
    #     enter_thread = threading.Thread(target=wait_for_enter)
    #     enter_thread.start()
    #
    #     def callback(indata, outdata, frames, time, status):
    #         recorded_data.append(indata.copy())
    #         audio_callback(indata, outdata, frames, time, status)
    #
    #     # Start the stream for real-time recording and processing
    #     with sd.Stream(samplerate=samplerate, channels=2, callback=callback):
    #         print("Recording and processing in real-time. Press Enter to stop.")
    #         enter_thread.join()  # Wait until Enter is pressed
    #
    #     # Convert the list of arrays to a single array
    #     recorded_data = np.concatenate(recorded_data, axis=0)
    #
    #     # Process the recorded data for plotting purposes
    #     t = np.arange(len(recorded_data)) / samplerate
    #     save_wav('recording.wav', recorded_data, samplerate)
    #     plt.figure()
    #     plt.subplot(3, 1, 1)
    #     plt.plot(np.fft.fftfreq(len(t), 1 / samplerate), np.abs(np.fft.fft(recorded_data[:, 0])))
    #     plt.title('Original Baseband Signal')
    #     plt.xlabel('Frequency (Hz)')
    #     plt.ylabel('Amplitude')
    #
    #     ssb_signal = ssb_modulate(recorded_data[:, 0], carrier_freq, samplerate)
    #     plt.subplot(3, 1, 2)
    #     plt.plot(np.fft.fftfreq(len(t), 1 / samplerate), np.abs(np.fft.fft(ssb_signal)))
    #     plt.title('SSB Modulated Signal (Upper Sideband)')
    #     plt.xlabel('Frequency (Hz)')
    #     plt.ylabel('Amplitude')
    #
    #     recovered_signal = ssb_demodulate(ssb_signal, carrier_freq, samplerate)
    #     save_wav('recording_modulated.wav', recovered_signal, samplerate)
    #     plt.subplot(3, 1, 3)
    #     plt.plot(np.fft.fftfreq(len(t), 1 / samplerate), np.abs(np.fft.fft(recovered_signal)))
    #     plt.title('Demodulated Signal')
    #     plt.xlabel('Frequency (Hz)')
    #     plt.ylabel('Amplitude')
    #
    #     plt.tight_layout()
    #     plt.show()
    #
    #     return recovered_signal

    else:
        print("Invalid mode or filename not provided for file mode.")

