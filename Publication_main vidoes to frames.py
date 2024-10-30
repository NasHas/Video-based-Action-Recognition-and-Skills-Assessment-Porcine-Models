'"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'
'### The following script is made as part of the study #Video-based robotic surgical action recognition and skills assessment on porcine models using deep learning###'
'### The code is open-source. However, when using the code, please make a reference to our paper and repository.""""""""""""""""""""""""""""""""""""""""""""""""""""""'
'"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'

'''
This code uses the name and local placement of a video to convert a video, using a predefined frame rate, into individual images placed in a folder of the same name as the original video + "Frames"
All frames are also placed in an "all-data" folder (which is used to load data from later).

**** NOTE ****
It can be an advantage to place all videos into one folder, so that one dosnt have to redefine the "LocalPlacement" for each day of data collection. Further if one does that,
the code can be modified to use the "convert" function iteratively by adding code that extracts the names of the videos in this colletive folder and then uses the convert function on that name.
'''
import cv2
import numpy as np
import os
import math


def convert(Prename, LocalPlacement, AllData_folder, framerate):
    file = LocalPlacement + Prename + '.mp4' # eller mp4 afhaengig af video
# Saves in another folder in the local place - but I've changed it to the allfolder
    folder = AllData_folder + Prename + 'Frames'

# Playing video from file:
    cap = cv2.VideoCapture(file)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    start_frame_number = 0
    currentFrame = 1

    videolength1 = int(frame_count//framerate)
    videolength = math.floor(videolength1 / 2) * 2 #We need to round down, otherwise the videolength sometimes dont add up and produces an error

    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
    except OSError:
        print ('Error: Creating directory of data')



    for i in range(videolength):
        start_frame_number += framerate

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
        success, frame = cap.read()

    # Saves image of the current frame in jpg file
        name = folder + '/' + Prename + ' frame' + str(currentFrame) + '.jpg'
        print('Creating...' + name)
        cv2.imwrite(name, frame)
        #Also save to "all data"- folder
        name_allData = AllData_folder + '/' + Prename + ' frame' + str(currentFrame) + '.jpg'
        cv2.imwrite(name_allData, frame)

    # To stop duplicate images
        currentFrame += 1

##### Porcine #####

convert('Name of file', r'PATH TO FILE', r'PATH TO FOLDER TO SAVE TO', FPS OF VIDEO)
# EXAMPLE: convert('Dissection colon', r'C:\Videos\COLON/', r'C:\AllFramesCOLON/', 30)
