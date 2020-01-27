import cv2
import numpy as np

# must be the same as camera resolutions 
frameWidth = 640
frameHeight = 480
haar_cascade_face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# 0 - is web camera
cap = cv2.VideoCapture(0)
# resizing
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)

while True:
    success, img = cap.read()
   
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    faces_rects = haar_cascade_face.detectMultiScale(imgGray, scaleFactor = 1.2, minNeighbors = 5)

    for (x,y,w,h) in faces_rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Faces", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
