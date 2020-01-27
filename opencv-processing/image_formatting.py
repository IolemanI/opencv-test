import cv2
import numpy as np

# 5 x 5 arrays of "1"
kernel = np.ones((5, 5), np.uint8)

path = "resources/image.jpg"
img = cv2.imread(path)
cv2.imshow("Image", img)

# grayscale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("ImageGray", imgGray)

# blurred
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
cv2.imshow("ImageBlur", imgBlur)

# canny (black with white edges)
imgCanny = cv2.Canny(imgBlur, 50, 50)
cv2.imshow("ImageCanny", imgCanny)

# a picture made of squares
imgDilation = cv2.dilate(imgCanny, kernel, iterations=1)
cv2.imshow("ImageDilation", imgDilation)

# just look the result :)
imgEroded = cv2.erode(imgDilation, kernel, iterations=1)
cv2.imshow("ImageEroded", imgEroded)

print(img.shape)
# resizing 
width, height = 400, 400
imgResize = cv2.resize(img, (width, height))
cv2.imshow("ImgResize", imgResize)

# crop
# img [height, width]
imgCropped = img[0:300, 0:900]
cv2.imshow("imgCropped", imgCropped)

cv2.waitKey(0)