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
"""מכניסים קובץ wav כאות כניסה אחר כך בודקים איזה סוג מסנן המשתמש רוצה
   בהתאם לבחירה שלו לפי הקובץ config מכניסים את שאר הפרמטרים הרלוונטים כולל הפקטור,  סוג הסינון(אם מדובר באינטרפולציה).
    ומפעילים את הפונקציה המבוקשת של הסינון ובנוסף  מעדכנים את הקלט של הפונקציה הבאה להיות הפלט של מה שיצא
  אם נבחר משהו שלא מתאים (אינטרפולציה או דצימציה ) מחזיר הודעת שגיאה ומאתחל את הקלט לאות הבא להיות האות המקורי שהוכנס 
"""
#מגדירים משתנה עם שם הקובץ
input_filename = "about_time.wav"
if(config['DI']['decimation_or_interpolation']=='decimation'):
    #בוחרים פקטור דצימציה
    decimation_factor = config['DI']['decimation_factor']
    input_to_ssb =decimation_factor
    # שולחים לפונקציה
    decimated_signal = decimate(input_filename, decimation_factor)
elif(config['DI']['decimation_or_interpolation']=='interpolation'):
    #לאינטרפולציה מכניסים את הפקטור של פי כמה להגדיל את מספר הדגימות
    interpolation_factor = config['DI']['interpolation_factor']
    #בוחרים באיזה מסנן להשתמש
    filter_type = config['DI']['interpolation_filter_type']
    #מכניסים הכל לתוך הפונקציה ובהצלחה
    interpolate_signal = interpolate(input_filename1, interpolation_factor, filter_type)
    input_to_ssb=interpolate_signal
else:
    print ("eror, you didnt choose the corect filter")
    print("the output stays as the input")
    input_to_ssb=input_filename

""" לפי מה שהמשתמש הגדיר בconfig בוחרים מצב של הssb ולפי זה מפעילים את הפונקציה עם הפרמטרים המתאימים 
אם לא נבחר מצב תקין משאיר את הפלט כמו הקלט"""
if(config[SSB]['ssb_mode']=='file'):
    ssb_transmittion = SSB(mode='file', file=input_to_ssb)
elif(config[SSB]['ssb_mode']=='live'):
    ssb_transmittion = SSB(mode='live', file=input_to_ssb)
else:
    print ("eror, you didnt choose the corect mode")
    print("the output stays as the input")
    ssb_transmittion=input_to_ssb
#בהערה יש דוגמא לכך ששירשרתי את הפונקציה עם עצמה כדי לראות שעובד
#interpolate_signal2 = interpolate(interpolate_signal, interpolation_factor, filter_type)
#interpolate_signal3 = interpolate(interpolate_signal2, interpolation_factor, filter_type)

