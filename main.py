# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print("hello apple")  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

#ido leibowitz
"""
conection decimation and interpolation
and transmission and recption ssb
"""
""""""
#בוחרים פקטור דצימציה
decimation_factor = 4
#מגדירים משתנה עם שם הקובץ
input_filename1 = "about_time.wav"
#שולחים לפונקציה
decimated_signal1 = decimate(input_filename1, decimation_factor)
#עוד נסיון עם קובץ אחר
input_filename2 = "activity_unproductive.wav"
decimated_signal2 = decimate(input_filename2, decimation_factor)
#לאינטרפולציה מכניסים את הפקטור של פי כמה להגדיל את מספר הדגימות
interpolation_factor = 5
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