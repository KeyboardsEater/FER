#import modules


#extract frames from video
#label frame by changing file name
#sort frames by labels

import os
import pandas as pd
import xlrd
import cv2
import dlib
import datetime
import time
import math

# cap.read(),cap.get(propid),cap.set(propid,values),cv2.waitKey() control video play speed

#read in data
file = 'annotation.xlsx'
df = pd.read_excel(file)
print(df.columns)
column = ['start', 'end', 'basic models', '2D model']
values = df[column].values #return a high dimensional shuzu

#create folders
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotions_2D =['sad', 'angry', 'disgust', 'dislike', 'fear', 'sleepy', 'bored',
              'neutral', 'intense', 'surprise', 'calm', 'relaxed', 'joyful', 'satisfied', 'happy', 'excited']
video_path = '../'

if 'video_frames' not in os.listdir('../'):#search file in the path
    os.mkdir('video_frames')#create file
for emotion in emotions:
    newdir = '../video_frames/' + emotion
    if emotion not in os.listdir('../video_frames/'):
        os.mkdir(newdir)

vidcap = cv2.VideoCapture('../fullvideo.mp4')
success = True
framerate = vidcap.get(5)
for i in values:
    count = 0
    pathOut = '../video_frames/' + i[2]
    start_time_ms = (i[0].hour * 3600 + i[0].minute * 60 + i[0].second) * 1000
    stop_time_ms = (i[1].hour * 3600 + i[1].minute * 60 + i[1].second) * 1000
    while success and vidcap.get(cv2.CAP_PROP_POS_MSEC) > start_time_ms:
        success, image = vidcap.read()#return a bool
    while success and vidcap.get(cv2.CAP_PROP_POS_MSEC) <= stop_time_ms:
        success, image = vidcap.read()
        if success == True:
            # pic size,quality
            cv2.imwrite(os.path.join(pathOut, "{:d}.jpg".format(count)), image)
            count += 1
        else:
            break

#read in video
