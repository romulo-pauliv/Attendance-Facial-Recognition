# +--------------------------------------------------------------------------------------------------------------------+
# |                                                                                              AttendanceProject.py ||
# |                                                                                             Author:Pauliv, Rômulo ||
# |                                                                                      Copyright 2022, RP11.AI Ltd. ||
# |                                                                                             https://(address.com) ||
# |                                                                                                      Version 1.0  ||
# +--------------------------------------------------------------------------------------------------------------------+

# imports -------------------------------------------------------------------------------------------------------------|
import colorama
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
import time
# install dlib in prompt pip install dlib==19.18.0
# ---------------------------------------------------------------------------------------------------------------------|

dirVideo = 0


def log(lognm):
    if lognm == 1:
        print(colorama.Fore.LIGHTCYAN_EX +
              "|------------------- ATTENDANCE BY FACIAL RECOGNITION -------------------| " +
              colorama.Style.RESET_ALL)
    if lognm == 2:
        print("\n \n")
        print(colorama.Fore.RED + "+------------------------------------------------------------------------+")
        print("|| EXCEPTION - FOLDER NOT FOUND                                          +")
        print("|| The system did not find a folder named 'imagesAttendance'             +")
        print("|| " + colorama.Fore.YELLOW + "Create a folder with that name include the necessary images           " +
              colorama.Fore.RED + "+")
        print("+------------------------------------------------------------------------+" + colorama.Style.RESET_ALL)
        quit()
    if lognm == 3:
        print(colorama.Fore.LIGHTCYAN_EX + "Process 1 |---------------------------------------------",
              colorama.Fore.MAGENTA + "READING PICTURES" + colorama.Fore.LIGHTCYAN_EX + "|" +
              colorama.Style.RESET_ALL)
    if lognm == 4:
        print(colorama.Fore.LIGHTCYAN_EX + "Process 2 |----------------------------------------------",
              colorama.Fore.MAGENTA + "ENCODE PICTURES" + colorama.Fore.LIGHTCYAN_EX + "|" +
              colorama.Style.RESET_ALL)
    if lognm == 5:
        print(colorama.Fore.LIGHTCYAN_EX + "Process 3 |-------------------------------------------",
              colorama.Fore.LIGHTGREEN_EX + "READING VIDEO FILE" + colorama.Fore.LIGHTCYAN_EX + "|" +
              colorama.Style.RESET_ALL)
    if lognm == 6:
        print(colorama.Fore.WHITE + "Webcam or local camera -------|" +
              colorama.Fore.MAGENTA + " DIR: " + str(dirVideo) + colorama.Style.RESET_ALL)
    if lognm == 7:
        print(colorama.Fore.WHITE + "Video DIR: " + colorama.Fore.MAGENTA + str(dirVideo) + colorama.Style.RESET_ALL)
    if lognm == 8:
        print(colorama.Fore.WHITE + "||||||||||||||||||| " +
              colorama.Fore.YELLOW + "STARTING FRAME CAPTURE AND LOOPING" +
              colorama.Fore.WHITE + " |||||||||||||||||||" +
              colorama.Style.RESET_ALL)


def find_encodings(images_list, images_names):                  # Function to encode images
    encodeList = []                                             # List to allocate the encoded images
    for i, img_item in enumerate(images_list):                  # Encode each images in the list
        img_item = cv2.cvtColor(img_item, cv2.COLOR_BGR2RGB)    # Convert image from BGR to RGB
        encode = face_recognition.face_encodings(img_item)[0]   # Encode the image
        encodeList.append(encode)                               # Add the encoded images to the list
        print(colorama.Fore.MAGENTA + str(images_names[i]) +
              colorama.Fore.WHITE + " ---------| " +
              colorama.Fore.GREEN + "ENCODED" + colorama.Style.RESET_ALL)
    return encodeList                                           # return to list


def mark_attendance(face_name):                             # Function to upload to a csv file
    with open('Attendance.csv', 'r+') as f:                 # Open the file and edit it (r+)
        myDataList = f.readline()                           # Read lines from the file and assign it to the variable
        nameList = []                                       # Create list and to concatenate
        for line in myDataList:
            entry = line.split(",")
            nameList.append(entry[0])
        if face_name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{face_name}, {dtString}')


class ExceptionPass(object):
    pass


log(lognm=1)
path = 'imagesAttendance'      # Name of the folder will be used to upload images
images = []                    # List to upload images
classNames = []                # List that will be uploaded file names
myList = []                    # List for files
try:
    myList = os.listdir(path)  # Convert the files inside the indicated folder to a list
except ExceptionPass():
    log(lognm=2)

log(lognm=3)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')             # Read image files
    images.append(curImg)                           # add to "images" list
    classNames.append(os.path.splitext(cl)[0])      # add the files names in the classNames list
    print(colorama.Fore.MAGENTA + str(os.path.splitext(cl)[0]) +
          colorama.Fore.WHITE + " ---------| " + colorama.Fore.YELLOW + str(f'{path}/{cl}'))

log(lognm=4)
encodeListKnown = find_encodings(images_list=images,
                                 images_names=classNames)  # Run and assign a variable to the encoder

log(lognm=5)
log(lognm=6) if type(dirVideo) == int else log(lognm=7)
cap = cv2.VideoCapture(dirVideo)
log(lognm=8)

# Process time:
TP_fullAlgo = []
TP_encodeFrame = []

while True:
    TP_fullAlgo_initial = time.time()
    success, img = cap.read()
    imgS = cv2.resize(src=img, dsize=(0, 0), dst=None, fx=0.25, fy=0.25, interpolation=None)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # SLOW PROCESSING - FACE_ENCODINGS
    TP_encodeFrame_initial = time.time()
    faceCurFrame = face_recognition.face_locations(imgS, model='hog')  # Find face in frame
    encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)  # Encode the face that faceCurName found
    TP_encodeFrame_final = time.time()
    for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):  # Distribute the encoder and location of the face
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)  # Compare face in video with all encoded
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  # Mathematical distance from face-coded
        matchIndex = np.argmin(faceDis)  # Returns the indices of minimum values along an axis
        if matches[matchIndex]:  # If there is comparison with the shortest distance (Boolean)
            name = classNames[matchIndex].upper()  # Find the name in the classNames list
            y1, x2, y2, x1 = faceLoc  # arrange face box coordinates (pt1, pt2)
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Resize cv2.resize values to actual size
            cv2.rectangle(img=img,  # plot the rectangle on the image
                          pt1=(x1, y1),
                          pt2=(x2, y2),
                          color=(0, 255, 0),
                          thickness=1)
            cv2.putText(img=img,  # plot the text on the image
                        text="{a} {b}".format(a=name, b=faceDis[matchIndex]),
                        org=(x1 + 4, y2 + 12),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.3,
                        color=(255, 255, 255),
                        thickness=1)
            mark_attendance(face_name=name)

        if not matches[matchIndex]:  # If you don't hear facial recognition
            y1i, x2i, y2i, x1i = faceLoc
            y1i, x2i, y2i, x1i = y1i * 4, x2i * 4, y2i * 4, x1i * 4
            cv2.rectangle(img=img,  # plot the rectangle on the image
                          pt1=(x1i, y1i),
                          pt2=(x2i, y2i),
                          color=(0, 0, 255),
                          thickness=2)
            cv2.putText(img=img,  # plot the text on the image
                        text="Unknown - ALERT",
                        org=(x1i + 4, y2i + 12),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 0, 255),
                        thickness=1)
            mark_attendance(face_name="Unknown - ALERT")

    cv2.imshow("ATTENDANCE - FACIAL RECOGNITION", img)
    TP_fullAlgo_final = time.time()

    TP_fullAlgo.append((TP_fullAlgo_final - TP_fullAlgo_initial))
    TP_encodeFrame.append((TP_encodeFrame_final - TP_encodeFrame_initial))

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
print(colorama.Fore.WHITE + "Loop processing time:   " +
      colorama.Fore.YELLOW + str(round(sum(TP_fullAlgo)/len(TP_fullAlgo), 4)) + colorama.Fore.WHITE + " s/frame" +
      colorama.Style.RESET_ALL)
print(colorama.Fore.WHITE + "Encode processing time: " +
      colorama.Fore.YELLOW + str(round(sum(TP_encodeFrame)/len(TP_encodeFrame), 4)) + colorama.Fore.WHITE + " s/frame" +
      colorama.Style.RESET_ALL)
