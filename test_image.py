import cv2
import numpy

img = cv2.imread("resources/image.jpg")

cv2.imshow("Image", img)

cv2.waitKey(0)