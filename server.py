import json
from flask import Flask
from flask import request
from flask import Response
import jsonpickle
import numpy as np
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    return Response(json.dumps({ "message": "Ok" }), status=200, mimetype='application/json')


@app.route('/image', methods = ['POST'])
def get_campaigns():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # do some fancy processing here....
    blur = cv2.Laplacian(img, cv2.CV_64F).var()
    cv2.imwrite(str(int(blur)) + ".jpg", img)

    # build a response dict to send back to client
    response = { 'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0]) }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == "__main__":
    app.run()
