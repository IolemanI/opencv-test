import json
from flask import Flask
from flask import request
from flask import Response
import jsonpickle
import numpy as np
import cv2
import pickle
import time

app = Flask(__name__)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch('resources/openface_nn4.small2.v1.t7')

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open('resources/output/recognizer.pickle', "rb").read())
le = pickle.loads(open('resources/output/le.pickle', "rb").read())

@app.route('/')
def index():
    return Response(json.dumps({ "message": "Ok" }), status=200, mimetype='application/json')


@app.route('/image', methods = ['POST'])
def get_campaigns():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    face = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # construct a blob for the face ROI, then pass the blob
    # through our face embedding model to obtain the 128-d
    # quantification of the face
    faceBlob = cv2.dnn.blobFromImage(
        face, 
        1.0 / 255, 
        (96, 96),
        (0, 0, 0), 
        swapRB=True, 
        crop=False
    )
    embedder.setInput(faceBlob)
    vec = embedder.forward()

    # perform classification to recognize the face
    preds = recognizer.predict_proba(vec)[0]
    j = np.argmax(preds)
    proba = preds[j]
    name = le.classes_[j]

    # uncoment this to save faces to disk
    # blur = cv2.Laplacian(face, cv2.CV_64F).var()
    # cv2.imwrite(f'{name}_{round(proba * 100, 2)}.jpg', face)

    # build a response dict to send back to client
    response = { 
        'image': {
            'size': '{}x{}'.format(face.shape[1], face.shape[0]),
        },
        'face': {
            'name': "{}".format(name),
            'confidence': float(proba)
        }
    }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == "__main__":
    app.run()
