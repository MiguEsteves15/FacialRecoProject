import cv2
import numpy as np
import face_recognition
import os
from PlayerOfSLButils import *

path = 'ImagesPlantel'
images = []
playerNames = []
list = os.listdir(path)

for name in list:
    img = cv2.imread(f'{path}/{name}')
    images.append(img)
    playerNames.append(os.path.splitext(name)[0])

print(playerNames)
KnownEncodes = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    FacesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, FacesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,FacesCurFrame):
        matches = face_recognition.compare_faces(KnownEncodes,encodeFace)
        faceDis = face_recognition.face_distance(KnownEncodes,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            print(playerNames[matchIndex])
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = 4*y1, 4*x2, 4*y2, 4*x1
            cv2.rectangle(img,(x1,y1),(x2,y2), (0, 255, 0), 5)
            nome = playerNames[matchIndex]
            font = cv2.FONT_HERSHEY_COMPLEX
            scale = 1
            thickness = 2
            (texto_largura, texto_altura), _ = cv2.getTextSize(nome, font, scale, thickness)
            cv2.rectangle(img, (x1, y2 - texto_altura - 10), (x1 + texto_largura + 12, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, nome, (x1 + 6, y2 - 6), font, scale, (0, 0, 0), thickness)
        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = 4 * y1, 4 * x2, 4 * y2, 4 * x1
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
            nome = 'UNKNOWN'
            font = cv2.FONT_HERSHEY_COMPLEX
            scale = 1
            thickness = 2
            (texto_largura, texto_altura), _ = cv2.getTextSize(nome, font, scale, thickness)
            cv2.rectangle(img, (x1, y2 - texto_altura - 10), (x1 + texto_largura + 12, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, nome, (x1 + 6, y2 - 6), font, scale, (0, 0, 0), thickness)

    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break