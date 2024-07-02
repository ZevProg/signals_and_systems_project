def read_wav(file_path):
    # Read the WAV file and return the header and audio data.
    with open(file_path, 'rb') as f:
        header = bytearray(f.read(44))  # Read the first 44 bytes (header)
        data = f.read()  # Read the rest of the audio data
    return header, data


def write_wav(file_path, header, data):
    # Write the header and audio data to a new WAV file.
    with open(file_path, 'wb') as f:
        f.write(header)  # Write the header
        f.write(data)  # Write the audio data


def change_speed(data, speed_factor):
    # Change the speed of the audio data by the given speed factor.
    sample_width = 2  # Each sample is 2 bytes (16 bits)
    num_samples = len(data) // sample_width  # Number of audio samples
    new_num_samples = int(num_samples / speed_factor)  # Adjust the number of samples

    new_data = bytearray()
    for i in range(new_num_samples):
        sample_index = int(i * speed_factor) * sample_width
        new_data.extend(data[sample_index:sample_index + sample_width])

    return bytes(new_data)


def update_header(header, data_length):
    # Update the header to reflect the new length of the audio data.
    chunk_size = 36 + data_length
    header[4:8] = chunk_size.to_bytes(4, byteorder='little')

    subchunk2_size = data_length
    header[40:44] = subchunk2_size.to_bytes(4, byteorder='little')

    return header


def process_wav(input_file, output_file, speed_factor):
    # Process the WAV file: read, change speed, update header, and write.
    header, data = read_wav(input_file)  # Read the original WAV file
    new_data = change_speed(data, speed_factor)  # Change the speed of the audio
    new_header = update_header(header, len(new_data))  # Update the header
    write_wav(output_file, new_header, new_data)  # Write the new WAV file


# Example usage:
input_file = 'input.wav'  # Enter wav file
output_file = 'output.wav'
speed_factor = 0.5  # Example: Use 0.5 to slow down, 2 to speed up

process_wav(input_file, output_file, speed_factor)
