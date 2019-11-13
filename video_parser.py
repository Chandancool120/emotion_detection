import cv2
import os
from os import listdir
import inspect
from os.path import isfile, join

def video_to_frames(video, emotion, file_index,video_number):
    vidcap = cv2.VideoCapture(video)
    totalimages = 0
    imageReactionStartNumber = 15
    while vidcap.isOpened():
        success, image = vidcap.read()
        index = file_index
        if success:

            if ((index - 31*video_number) /imageReactionStartNumber >1):
                filepath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
                filepath = filepath + '/' + str(emotion) + '/' + str(file_index) + '.png'
                cv2.imwrite(filepath, image)
            file_index+=1
            totalimages+=1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()
    return totalimages
 
dirs = [ '1_Neutral-Happiness','2_Neutral-Anger', '4_Neutral-Sadness', '3_Neutral-Fear','5_Neutral-Disgust','6_Neutral-Surprise' ]
print(dirs)
emotion =0
file_index =0

for f in dirs:
     onlyfiles = [join(f, file) for file in listdir(f)]
     video_number = 0
     for file in onlyfiles:
          rcount = video_to_frames(file, emotion, file_index,video_number)
          file_index+=rcount
          video_number += 1
     emotion+=1
     file_index=0
