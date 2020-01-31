from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import requests
import jsonpickle
from multiprocessing.dummy import Process

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-f', '--frame-rate', type=float, default=0.5,
                help='minimum frame rate to use less CPU')
ap.add_argument('-c', '--confidence', type=float, default=0.7,
                help='minimum probability to filter weak detections')

# SMALLER BLUR VALUE MEANS MORE BLURRINESS PRESENT
ap.add_argument('-b', '--blur', type=int, default=200,
                help='minimum blur to filter low quality')
ap.add_argument('-l', '--link', help='IP webcam link')
ap.add_argument('-d', '--delta', type=float, default=7, help='minimum pause after face recognition')
args = vars(ap.parse_args())

timeFlag = time.perf_counter()

# load our serialized face detector from disk
print('[INFO] loading face detector...')
detectorPath = 'resources/face_detection_model'
protoPath = os.path.sep.join([detectorPath, 'deploy.prototxt'])
modelPath = os.path.sep.join(
    [detectorPath, 'res10_300x300_ssd_iter_140000.caffemodel'])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# initialize the video stream, then allow the camera sensor to warm up
print('[INFO] starting video stream...')

videoLink = args['link']
if videoLink is None:
    print('[INFO] integrated webcam')
    vs = VideoStream(src=0).start()
else:
    print(f'[INFO] from link {videoLink}')
    # rtsp://192.168.1.104:8080/h264_ulaw.sdp - for IP Webcam Android app
    vs = VideoStream(src=videoLink).start()

time.sleep(2.0)
print('[INFO] frame rate is {}s'.format(args['frame_rate']))
print('[INFO] min confidence is {:.2f}%'.format(args['confidence'] * 100))
print('[INFO] min blur is {}'.format(args['blur']))


def sendShapshot(snapshot, callback):
    URL = 'http://127.0.0.1:5000/image'

    # prepare headers for http request
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}

    # encode image as jpeg
    _, img_encoded = cv2.imencode('.jpg', snapshot)

    # send http request with image and receive and decode response
    response = requests.post(URL, data=img_encoded.tostring(), headers=headers)
    decoded = jsonpickle.decode(response.text)
    # print('[DEBUG] decoded: ', decoded['face'])

    callback(decoded['face'])

def handlePass(face):
    name = face['name']
    confidence = round(face['confidence'] * 100, 2)
    
    if name == 'unknown':
        print('[DEBUG] denied: ', face['name'], confidence)
    else:
        # open the door, uter something etc.
        print('[DEBUG] passed: {} {}'.format(face['name'], confidence))



def getDetections(frame):
    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0), 
        swapRB=False, 
        crop=False
    )

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    confidence = detections[0, 0, 0, 2]
    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])

    return (confidence, box)

def drawBoundaries():
    # draw the bounding box of the face along with the associated probability
    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 1)

    confidenceText = '{:.2f}%'.format(confidence * 100)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(frame, confidenceText, (startX, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

# loop over frames from the video file stream
while True:
    time.sleep(args['frame_rate'])

    frame = vs.read()

    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    (confidence, box) = getDetections(frame)

    if confidence > args['confidence']:
        (startX, startY, endX, endY) = box.astype('int')

        face = frame[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]

        # ensure the face width and height are sufficiently large
        if fW < 20 or fH < 20:
            continue

        drawBoundaries()

        blur = cv2.Laplacian(face, cv2.CV_64F).var()

        delta = round(time.perf_counter() - timeFlag, 2)
        print(f'[DEBUG] Face detected, wait { round(args["delta"] - delta) }')
        if blur > args['blur'] and delta > args['delta']:
            timeFlag = time.perf_counter()
            print('[INFO] sending the snapshot: {}, {}'.format(blur, round(confidence * 100)))

            # sending request to the server for person recognition in parallel 
            Process(target=sendShapshot, args=(face, handlePass)).start()


    # show the output frame
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord('q'):
        break


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
