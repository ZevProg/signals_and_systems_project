# import modules.pdm2pcm_Noam_Tehila as Pdm2Pcm
# import modules.DC_removal as DCRemoval
# import modules.Voice_Activity_Detector_Einav_Avital as VAD
# import modules.Acoustic_Gain_Control as AGC
# import modules.Short_Time_Fourier_Transform as STFT
# import modules.noise_reduction_Tehila_Shira as NoiseReduction
# import modules.Pitch_Estimation_Naama_Shira as PitchEstimation
# import modules.slowing_Or_speeding_Speech_Hamles_Tzuf as SpeechSpeed
# import modules.decimation_and_interpolation_Dolev_Segev_Yuval as DI
# import modules.Transmition_And_Reception as SSB 
# import yaml

# with open('config.yml', 'r') as f:
#     config = yaml.safe_load(f)

# #Hadar Semama
# #Connection of pdm2pcm and dc removal

# pcm_sample_rate=config['Pdm2Pcm']['pcm_sample_rate']
# decimation_factor=config['Pdm2Pcm']['decimation_factor']
# order=config['Pdm2Pcm']['order']
# user_pdm_file_path =config['Pdm2Pcm']['user_pdm_file_path']

# #Convert the pdm file to a wav signal
# pdm_output=Pdm2Pcm(user_pdm_file_path)

# #Pdm2Pcm converts the pdm file received from the user into a wave signal
# #The wave signal enters as input to the dc removal model 
# cutoff_frequency=config['DCRemoval']['cutoff_frequency']
# numtaps=config['DCRemoval']['numtaps']

# #Removal of the DC part of the signal
# dc_removal_output=DC_Removal_filter(pdm_output) 

# #Aviya Nave

# #VAD confing
# frame_duration = config['VAD']['frame_duration']
# threshold = config['VAD']['threshold']
# smoothness = config['VAD']['smoothness']
# plot_graphs = config['VAD']['plot_graphs']
# remove_dc = config['VAD']['remove_dc']

# #DC to VAD
# #binary vector for speech detection (1 for speech 0 for noise)
# binary_vector  =  process_audio_file(dc_removal_output)

# #AGC confing
# frame_duration = config['AGC']['frame_duration']
# gain = config['AGC']['gain']

# #VAD to AGC
# #applying AGC based on voice activity detection (using VAD output) 
# agc_output =  vad_aware_agc_process(dc_removal_output, binary_vector)


# #ido leibowitz
# """
# conection decimation and interpolation
# and transmission and recption ssb
# """
# """מכניסים קובץ wav כאות כניסה אחר כך בודקים איזה סוג מסנן המשתמש רוצה
#    בהתאם לבחירה שלו לפי הקובץ config מכניסים את שאר הפרמטרים הרלוונטים כולל הפקטור,  סוג הסינון(אם מדובר באינטרפולציה).
#     ומפעילים את הפונקציה המבוקשת של הסינון ובנוסף  מעדכנים את הקלט של הפונקציה הבאה להיות הפלט של מה שיצא
#   אם נבחר משהו שלא מתאים (אינטרפולציה או דצימציה ) מחזיר הודעת שגיאה ומאתחל את הקלט לאות הבא להיות האות המקורי שהוכנס 
# """
# #מגדירים משתנה עם שם הקובץ
# input_filename = "about_time.wav"
# if(config['DI']['decimation_or_interpolation']=='decimation'):
#     #בוחרים פקטור דצימציה
#     decimation_factor = config['DI']['decimation_factor']
#     input_to_ssb =decimation_factor
#     # שולחים לפונקציה
#     decimated_signal = decimate(input_filename, decimation_factor)
# elif(config['DI']['decimation_or_interpolation']=='interpolation'):
#     #לאינטרפולציה מכניסים את הפקטור של פי כמה להגדיל את מספר הדגימות
#     interpolation_factor = config['DI']['interpolation_factor']
#     #בוחרים באיזה מסנן להשתמש
#     filter_type = config['DI']['interpolation_filter_type']
#     #מכניסים הכל לתוך הפונקציה ובהצלחה
#     interpolate_signal = interpolate(input_filename1, interpolation_factor, filter_type)
#     input_to_ssb=interpolate_signal
# else:
#     print ("eror, you didnt choose the corect filter")
#     print("the output stays as the input")
#     input_to_ssb=input_filename

# """ לפי מה שהמשתמש הגדיר בconfig בוחרים מצב של הssb ולפי זה מפעילים את הפונקציה עם הפרמטרים המתאימים 
# אם לא נבחר מצב תקין משאיר את הפלט כמו הקלט"""
# if(config[SSB]['ssb_mode']=='file'):
#     ssb_transmittion = SSB(mode='file', file=input_to_ssb)
# elif(config[SSB]['ssb_mode']=='live'):
#     ssb_transmittion = SSB(mode='live', file=input_to_ssb)
# else:
#     print ("eror, you didnt choose the corect mode")
#     print("the output stays as the input")
#     ssb_transmittion=input_to_ssb
# #בהערה יש דוגמא לכך ששירשרתי את הפונקציה עם עצמה כדי לראות שעובד
# #interpolate_signal2 = interpolate(interpolate_signal, interpolation_factor, filter_type)
# #interpolate_signal3 = interpolate(interpolate_signal2, interpolation_factor, filter_type)


# # Shahar Goldstein

# #Pitch Estimation
# audio_data = # The output of "Noise Reduction"
# pitch_output = PitchEstimation(audio_data)
# # Now We can plot the pitch if needed

# #SpeechSpeed
# speed_factor = config['SpeechSpeed']['speed_factor']
# input_wave_data = # The output of "Noise Reduction"
# new_speed_speech = SpeechSpeed(input_wave_data)


import yaml
import io
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
    # NoiseReduction.NoiseReduction(io.BytesIO(agc_output), noise_reduction_output, vad_output, frame_size, hop_size)
    NoiseReduction.NoiseReduction(vad_output, noise_reduction_output,io.BytesIO(agc_output) , frame_size, hop_size)

    # 6. Pitch Estimation
    pitch_output = PitchEstimation.process_wav_file_pitches(wave.open(noise_reduction_output, 'rb'))

    # 7. Speech Speed Modification
    speed_factor = config['SpeechSpeed']['speed_factor']
    with open(pitch_output, 'rb') as f:
        speech_speed_output = SpeechSpeed.process_wav_data(f.read(), speed_factor)

    # 8. Decimation and Interpolation
    di_mode = config['DI']['decimation_or_interpolation']
    if di_mode == 'decimation':
        di_factor = config['DI']['decimation_factor']
        di_output = DI.decimate(io.BytesIO(speech_speed_output), di_factor)
    elif di_mode == 'interpolation':
        di_factor = config['DI']['interpolation_factor']
        filter_type = config['DI']['interpolation_filter_type']
        di_output = DI.interpolate(io.BytesIO(speech_speed_output), di_factor, filter_type)
    else:
        raise ValueError(f"Invalid DI mode: {di_mode}")

    # 9. SSB Transmission and Reception
    ssb_mode = config['SSB']['ssb_mode']
    ssb_output = SSB.SSB(mode=ssb_mode, file=io.BytesIO(di_output))

    print(f"Final output saved to: {ssb_output}")

if __name__ == "__main__":
    main()