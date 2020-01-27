import cv2
import numpy as np
import utils

# init camera 
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def empty(a):
    pass

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 240)
cv2.createTrackbar("Hue min", "HSV",0,179,empty)
cv2.createTrackbar("Hue max", "HSV",179,179,empty)
cv2.createTrackbar("Sat min", "HSV",0,255,empty)
cv2.createTrackbar("Sat max", "HSV",255,255,empty)
cv2.createTrackbar("Value min", "HSV",0,255,empty)
cv2.createTrackbar("Value max", "HSV",255,255,empty)


while True:
    _, img = cap.read()
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("Hue min", "HSV")
    h_max = cv2.getTrackbarPos("Hue max", "HSV")
    s_min = cv2.getTrackbarPos("Sat min", "HSV")
    s_max = cv2.getTrackbarPos("Sat max", "HSV")
    v_min = cv2.getTrackbarPos("Value min", "HSV")
    v_max = cv2.getTrackbarPos("Value max", "HSV")
    
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)

    result = cv2.bitwise_and(img, img, mask = mask)

    cv2.imshow("Image Detection", utils.stackImages(0.3, [img, mask, result]))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()