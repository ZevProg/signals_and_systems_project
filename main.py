import modules.pdm2pcm_Noam_Tehila as Pdm2Pcm
import modules.DC_removal as DCRemoval
import modules.Voice_Activity_Detector_Einav_Avital as VAD
import modules.Acoustic_Gain_Control as AGC
import modules.Short_Time_Fourier_Transform as STFT
import modules.noise_reduction_Tehila_Shira as NoiseReduction
import modules.Pitch_Estimation_Naama_Shira as PitchEstimation
import modules.slowing_Or_speeding_Speech_Hamles_Tzuf as SpeechSpeed
import modules.decimation_and_interpolation_Dolev_Segev_Yuval as DI
import modules.Transmition_And_Reception as SSB 
import yaml

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

#Hadar Semama
#Connection of pdm2pcm and dc removal

#The address of the user's pdm file
pcm_sample_rate=config['Pdm2Pcm']['pcm_sample_rate']
decimation_factor=config['Pdm2Pcm']['decimation_factor']
order=config['Pdm2Pcm']['order']
user_pdm_file_path =config['Pdm2Pcm']['user_pdm_file_path']

#Convert the pdm file to a wav signal
pdm_output=Pdm2Pcm(user_pdm_file_path)

#Pdm2Pcm converts the pdm file received from the user into a wave signal
#The wave signal enters as input to the dc removal model 
cutoff_frequency=config['DCRemoval']['cutoff_frequency']
numtaps=config['DCRemoval']['numtaps']

#Removal of the DC part of the signal
dc_removal_output=DC_Removal_filter(pdm_output) 




#ido leibowitz
"""
conection decimation and interpolation
and transmission and recption ssb
"""
""""""
#בוחרים פקטור דצימציה
decimation_factor = config['DI']['decimation_factor']
#מגדירים משתנה עם שם הקובץ
input_filename1 = "about_time.wav"
#שולחים לפונקציה
decimated_signal1 = decimate(input_filename1, decimation_factor)
#עוד נסיון עם קובץ אחר
input_filename2 = "activity_unproductive.wav"
decimated_signal2 = decimate(input_filename2, decimation_factor)
#לאינטרפולציה מכניסים את הפקטור של פי כמה להגדיל את מספר הדגימות
interpolation_factor = config['DI']['interpolation_factor']
#בוחרים באיזה מסנן להשתמש
filter_type = 'shanon'
#מגדירים משתנה שמקבל את הקישור לקובץ אודיו
input_filename3 = "about_time.wav"
#מכניסים הכל לתוך הפונקציה ובהצלחה
interpolate_signal = interpolate(input_filename3, interpolation_factor, filter_type)
#בהערה יש דוגמא לכך ששירשרתי את הפונקציה עם עצמה כדי לראות שעובד
#interpolate_signal2 = interpolate(interpolate_signal, interpolation_factor, filter_type)
#interpolate_signal3 = interpolate(interpolate_signal2, interpolation_factor, filter_type)
#  עבור הפונקציה של ssb צריך לבחור אם רוציה filr או live ניתן לראות כי הקובץ שהכנסתי הוא קובץ שנוצר באינטרפולציה ולכן השירשור עובד בין הפונקציות
ssb_transmittion = SSB(mode='file',file=interpolate_signal)
#זה דומא להרצה של הלייב מה שאני הבנתי זה שאין משמעות לשם של הקובץ כל עוד  הוא חוקי כי הלייב לא משתמש בו באמת פשוט הם מוגדרים באותה פונקציה אז צריך לשלוח קובת חוקי
#ssb_transmittion = SSB(mode='live',file=interpolate_signal)
