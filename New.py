import numpy as np
import face_recognition
import os
from datetime import datetime
import cv2
import time

# from PIL import ImageGrab
typeDev = int(input("Enter The Device Type 0, 1 : "))

path = "Training_images"
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


newname = []
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f: 
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            i=0
            entry = line.split(',')
            nameList.append(entry[i][0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')
            i+=1

encodeListKnown = findEncodings(images)
print('Encoding Complete')
cap = cv2.VideoCapture(typeDev)
while True:
    success, img = cap.read()  
# img = captureScreen()
    imgS = cv2.resize(img,(360,360))
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].capitalize()
            if name not in newname:
                newname.append(name)
                markAttendance(name)
                print(newname)
                
    cv2.imshow('Webcam', imgS)
    cv2.waitKey(100)
    
