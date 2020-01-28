import cv2
import os
import numpy as np
import utils
import time

# must be the same as camera resolutions 
frameWidth = 1280
frameHeight = 720
myPath = 'data/images'
cameraNo = 0
cameraBrightness = 180
moduleVal = 10  # SAVE EVERY ITH FRAME TO AVOID REPETITION
minBlur = 200  # SMALLER VALUE MEANS MORE BLURRINESS PRESENT
grayImage = False # IMAGES SAVED COLORED OR GRAY
saveData = True   # SAVE DATA FLAG
showImage = False  # IMAGE DISPLAY FLAG

haar_cascade_face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# 0 - is web camera
cap = cv2.VideoCapture(0)
# resizing
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,cameraBrightness)

count = 0
countSave =0

def saveDataFunc():
    global countFolder
    countFolder = 0
    while os.path.exists( myPath+ str(countFolder)):
        countFolder += 1
    os.makedirs(myPath + str(countFolder))

if saveData:saveDataFunc()

while True:
    success, img = cap.read()
   
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    faces_rects = haar_cascade_face.detectMultiScale(imgGray, scaleFactor = 1.2, minNeighbors = 5)
    for (x,y,w,h) in faces_rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if grayImage:img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        if saveData:
            blur = cv2.Laplacian(img, cv2.CV_64F).var() 
            print(count % moduleVal == 0, blur, minBlur)
            if count % moduleVal == 0 and blur > minBlur:
                nowTime = time.time()
                cv2.imwrite(myPath + str(countFolder) +
                        '/' + str(countSave)+"_"+ str(int(blur))+"_"+str(nowTime)+".png", img[y:y+h, x:x+w])
                countSave += 1
            count += 1

        if showImage:
            cv2.imshow("Image", img)


    # cropped = [img]
    # for (x,y,w,h) in faces_rects:
    #     cropped.append(img[y:y+h, x:x+w])
    
    # StackedImages = utils.stackImages(0.3,(cropped))
    cv2.imshow("Faces", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()