import tempfile
import yaml
import io
import os
import wave
from modules import pdm2pcm_Noam_Tehila as Pdm2Pcm
from modules import DC_removal as DCRemoval
from modules import Voice_Activity_Detector_Einav_Avital as VAD
from modules import Acoustic_Gain_Control as AGC
from modules import noise_reduction_Tehila_Shira as NoiseReduction
from modules import Pitch_Estimation_Naama_Shira as PitchEstimation
from modules import slowing_Or_speeding_Speech_Hamles_Tzuf as SpeechSpeed
from modules import decimation_and_interpolation_Dolev_Segev_Yuval as DI
from modules import Transmition_And_Reception as SSB

def load_config():
    with open('config.yml', 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    # 1. PDM to PCM conversion
    pdm_file_path = config['Pdm2Pcm']['user_pdm_file_path']
    decimation_factor = config['Pdm2Pcm']['decimation_factor']
    order = config['Pdm2Pcm']['order']
    pcm_sample_rate = config['Pdm2Pcm']['pcm_sample_rate']
    pcm_output = Pdm2Pcm.Pdm2Pcm(pdm_file_path, decimation_factor, order, pcm_sample_rate)

    # 2. DC removal
    cutoff_frequency = config['DCRemoval']['cutoff_frequency']
    numtaps = config['DCRemoval']['numtaps']
    dc_removal_output = DCRemoval.DC_Removal_filter(io.BytesIO(pcm_output),cutoff_frequency, numtaps)

    # 3. Voice Activity Detection
    frame_duration = config['VAD']['frame_duration']
    threshold = config['VAD']['threshold']
    smoothness = config['VAD']['smoothness']
    remove_dc = config['VAD']['remove_dc']
    plot_graphs = config['VAD']['plot_graphs']
    vad_output = VAD.process_audio_file(dc_removal_output, frame_duration, threshold, smoothness, remove_dc, plot_graphs)

    # 4. Acoustic Gain Control
    frame_duration = config['AGC']['frame_duration']
    gain = config['AGC']['gain']
    agc_output = AGC.vad_aware_agc_process(dc_removal_output, vad_output, frame_duration, gain)

    # 5. Noise Reduction
    frame_size = config['NoiseReduction']['frame_size']
    hop_size = config['NoiseReduction']['hop_size']
    noise_reduction_output = 'noise_reduction_output.wav'
    NoiseReduction.NoiseReduction(io.BytesIO(agc_output), noise_reduction_output, vad_output, frame_size, hop_size)


    # 6. Pitch Estimation
    plot_pitch_estimation = config['PitchEstimation']['plot_pitch_estimation']
    pitch_output = PitchEstimation.process_wav_file_pitches(wave.open(noise_reduction_output, 'rb'), plot_pitch_estimation)

    # 7. Speech Speed Modification
    speed_factor = config['SpeechSpeed']['speed_factor']
    with open(noise_reduction_output, 'rb') as f:
        speech_speed_output = SpeechSpeed.process_wav_data(f.read(), speed_factor)

    # Save speech_speed_output to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_file.write(speech_speed_output)
        temp_file_path = temp_file.name

    # 8. Decimation and Interpolation
    di_mode = config['DI']['decimation_or_interpolation']
    if di_mode == 'decimation':
        di_factor = config['DI']['decimation_factor']
        di_output = DI.decimate(temp_file_path, di_factor)
    elif di_mode == 'interpolation':
        di_factor = config['DI']['interpolation_factor']
        filter_type = config['DI']['interpolation_filter_type']
        di_output = DI.interpolate(temp_file_path, di_factor, filter_type)
    else:
        raise ValueError(f"Invalid DI mode: {di_mode}")

    # Clean up the temporary file
    os.unlink(temp_file_path)

    # 9. SSB Transmission and Reception
    ssb_mode = config['SSB']['ssb_mode']
    with open(di_output, 'rb') as f:
        di_data = f.read()
    ssb_output = SSB.SSB(mode=ssb_mode, file=io.BytesIO(di_data))


    print(f"Final output saved to: {ssb_output}")

if __name__ == "__main__":
    main()
    # remember to install the required packages
    # pip install -r requirements.txt