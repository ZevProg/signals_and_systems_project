import struct
import io

def process_wav_data(input_wav_data, speed_factor):
    """
    Process the WAV file data: read, change speed, and return modified data.
    
    :param input_wav_data: bytes object containing the entire WAV file
    :param speed_factor: float, speed factor to apply (e.g., 0.5 to slow down, 2 to speed up)
    :return: bytes object containing the modified WAV file
    """
    def read_wav_header(data):
        """Read the WAV file header from the data."""
        if data[:4] != b'RIFF' or data[8:12] != b'WAVE':
            raise ValueError("Not a valid WAV file")

        channels = struct.unpack_from('<H', data, 22)[0]
        sample_rate = struct.unpack_from('<I', data, 24)[0]
        bits_per_sample = struct.unpack_from('<H', data, 34)[0]

        return channels, sample_rate, bits_per_sample

    def write_wav_header(f, channels, sample_rate, bits_per_sample, data_size):
        """Write a valid WAV header to the file-like object."""
        f.write(b'RIFF')
        f.write(struct.pack('<I', data_size + 36))  # File size - 8
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))  # Size of fmt chunk
        f.write(struct.pack('<H', 1))  # Audio format (1 for PCM)
        f.write(struct.pack('<H', channels))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * channels * bits_per_sample // 8))  # Byte rate
        f.write(struct.pack('<H', channels * bits_per_sample // 8))  # Block align
        f.write(struct.pack('<H', bits_per_sample))
        f.write(b'data')
        f.write(struct.pack('<I', data_size))

    def change_speed(data, speed_factor, sample_width):
        """Change the speed of the audio data by the given speed factor."""
        num_samples = len(data) // sample_width
        new_num_samples = int(num_samples / speed_factor)
        new_data = bytearray()
        for i in range(new_num_samples):
            sample_index = int(i * speed_factor) * sample_width
            if sample_index + sample_width <= len(data):
                new_data.extend(data[sample_index:sample_index + sample_width])
        return bytes(new_data)

    try:
        # Edge case: Check if speed_factor is valid
        if speed_factor <= 0:
            raise ValueError("Speed factor must be greater than 0.")

        channels, sample_rate, bits_per_sample = read_wav_header(input_wav_data)
        sample_width = bits_per_sample // 8

        # Process the audio data
        audio_data = input_wav_data[44:]  # Skip header
        processed_data = change_speed(audio_data, speed_factor, sample_width)

        # Edge case: Check if processing resulted in empty data
        if not processed_data:
            raise ValueError("Processing resulted in empty audio data.")

        # Create a new in-memory file to store the result
        output_buffer = io.BytesIO()

        # Write the new header
        write_wav_header(output_buffer, channels, sample_rate, bits_per_sample, len(processed_data))

        # Write the processed audio data
        output_buffer.write(processed_data)

        # Get the final result as bytes
        result = output_buffer.getvalue()

        return result
    except Exception as e:
        print(f"Error processing WAV data: {e}")
        return None

# Example usage:
# with open('input.wav', 'rb') as file:
#     input_wav_data = file.read()
# 
# speed_factor = 1.5
# output_wav_data = process_wav_data(input_wav_data, speed_factor)
# 
# if output_wav_data:
#     with open('output.wav', 'wb') as file:
#         file.write(output_wav_data)
