import struct


def read_wav_header(file_path):
    """Read the WAV file header and return the header information."""
    with open(file_path, 'rb') as f:
        header = f.read(44)

        if header[:4] != b'RIFF' or header[8:12] != b'WAVE':
            raise ValueError("Not a valid WAV file")

        channels = struct.unpack_from('<H', header, 22)[0]
        sample_rate = struct.unpack_from('<I', header, 24)[0]
        bits_per_sample = struct.unpack_from('<H', header, 34)[0]

    return header, channels, sample_rate, bits_per_sample


def write_wav_header(f, channels, sample_rate, bits_per_sample, data_size):
    """Write a valid WAV header."""
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


def process_wav(input_file, output_file, speed_factor):
    """Process the WAV file: read, change speed, and write."""
    try:
        header, channels, sample_rate, bits_per_sample = read_wav_header(input_file)
        sample_width = bits_per_sample // 8

        with open(input_file, 'rb') as in_file:
            in_file.seek(44)  # Skip header

            # Read and process data in chunks
            chunk_size = 1024 * sample_width
            with open(output_file, 'wb') as out_file:
                # Write a placeholder header
                write_wav_header(out_file, channels, sample_rate, bits_per_sample, 0)

                total_written = 0
                while True:
                    chunk = in_file.read(chunk_size)
                    if not chunk:
                        break
                    processed_chunk = change_speed(chunk, speed_factor, sample_width)
                    out_file.write(processed_chunk)
                    total_written += len(processed_chunk)

                # Go back and write the correct header
                out_file.seek(0)
                write_wav_header(out_file, channels, sample_rate, bits_per_sample, total_written)

        print(f"Processed {input_file} with speed factor {speed_factor} and saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error processing WAV file: {e}")
        return False


# Example usage
while True:
    input_file = 'input2.wav'
    output_file = 'output.wav'
    try:
        speed_factor = float(input("Enter speed factor (e.g., 0.5 to slow down, 2 to speed up): "))
        if speed_factor <= 0:
            raise ValueError("Speed factor must be greater than 0.")
    except ValueError as e:
        print(f"Invalid input: {e}")
        continue
    success = process_wav(input_file, output_file, speed_factor)
    if success:
        break
    else:
        print("Please try again.")
