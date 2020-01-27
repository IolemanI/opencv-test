import cv2
import numpy as np

img = cv2.imread('resources/image.jpg')
print(img.shape)

img = cv2.resize(img, (0, 0), None, 0.5, 0.5)

hor = np.hstack((img, img))
ver = np.vstack((img, img))

cv2.imshow("v Image", ver)
cv2.imshow("h Image", hor)

cv2.waitKey(0)