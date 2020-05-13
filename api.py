from __future__ import print_function
import tensorflow as tf
import os

if os.environ.get('GPU') is not None:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.25
    if os.environ.get('GPU_FRACTION') is not None:
        config.gpu_options.per_process_gpu_memory_fraction = float(os.environ.get('GPU_FRACTION'))
    session = tf.Session(config=config)

from pathlib import Path
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
import tensorflow.contrib.util as tf_contrib_util
import datetime
import os.path
import json
from flask import Flask, request, send_from_directory, jsonify, Response
import urllib.request
from pprint import pprint
from base64 import b64encode, b64decode
from flask_cors import CORS
import httplib2
from json import encoder
from functools import wraps
import time

encoder.FLOAT_REPR = lambda o: format(o, '.2f')
COLOR = (255,92,122)
font = cv2.FONT_HERSHEY_SIMPLEX

confThreshold = 0.5                      # Confidence threshold default
nmsThreshold = 0.5
margin = 0.4

weights_file = "weights.hdf5"

with open('config.json') as f:
    config = json.load(f)

print("Using configs:")
pprint(config)

basic_auth_username = config["basic_auth_username"] # basic auth username
basic_auth_password = config["basic_auth_password"]

if os.environ.get('BASIC_AUTH_USERNAME') is not None:
    basic_auth_username = os.environ.get('BASIC_AUTH_USERNAME')

if os.environ.get('BASIC_AUTH_PASSWORD') is not None:
    basic_auth_password = os.environ.get('BASIC_AUTH_PASSWORD')

img_size = 64
width = 8
depth = 16
margin = 0.4
k = width
weight_file = "weights.hdf5"

detector_fast = dlib.get_frontal_face_detector()
detector_accurate = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
detector = None

model_loaded = False
model = None

def check_auth(username, password):
    """This function is called to check if a username /
    password combination is valid.
    """
    return username == basic_auth_username and password == basic_auth_password

def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
    'Could not verify your access level for that URL.\n'
    'You have to login with proper credentials', 401,
    {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

def downloadFile(URL=None):
    h = httplib2.Http(".cache")
    resp, content = h.request(URL, "GET")
    return content

def url_ok(url=""):
    if (url is None):
        return False
    return len(url) != 0

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

app = Flask(__name__, static_url_path='/root/face')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)

@app.route("/predict", methods=['POST'])
@requires_auth
def predict():
    """
    Predict Handler
    """
    try:
        json_data = json.loads(request.data.decode("utf-8"))
        url = ""
        base64 = ""
        draw = False;
        return_image = False;
        get_attributes = True;
        global model_loaded
        global model
        global detector_accurate
        global detector_fast
        global detector
        face_detector = "fast"

        detector = detector_fast

        global nmsThreshold
        global confThreshold

        if 'url' in json_data:
            url = json_data["url"]
        if 'base64' in json_data:
            base64 = json_data["base64"]
        if 'draw' in json_data:
            draw = json_data["draw"]
        if 'return_image' in json_data:
            return_image = json_data["return_image"]
        if 'get_attributes' in json_data:
            get_attributes = json_data["get_attributes"]
        if 'nms' in json_data:
            nmsThreshold = json_data["nms"]
        if 'confidence' in json_data:
            confThreshold = json_data["confidence"]
        if 'face_detector' in json_data:
            face_detector = json_data["face_detector"]
            if face_detector == "accurate":
                detector = detector_accurate

        if (not url_ok(url) and len(base64) == 0):
            url = request.form.get('url')
        elif (url_ok(url)):
            photo = downloadFile(url)
            frame = cv2.imdecode(np.fromstring(photo, np.uint8), 1)
        elif (len(request.files) > 0):
            photo = request.files.get('image')
            frame = cv2.imdecode(np.fromstring(photo.read(), np.uint8), 1)
        elif (len(base64) > 0):
            photo = base64
            type = "base64"
            input = photo.split(',')[1]
            nparr = np.fromstring(b64decode(input), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            photo = request.data
            # photo = cv2.imdecode(np.fromstring(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
            type = "string"

        height, width, chan = frame.shape

        start_time = datetime.datetime.now()
        img_h, img_w, _ = np.shape(frame)
        detected = detector(frame, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))
        _faces = [];

        if model_loaded == False:
            model = WideResNet(img_size, depth=depth, k=k)()
            model.load_weights(weight_file)
            model_loaded = True

        if len(detected) > 0:
            for i, d in enumerate(detected):
                if hasattr(d, 'rect'):
                    x1, y1, x2, y2, w, h, conf = d.rect.left(), d.rect.top(), d.rect.right() + 1, d.rect.bottom() + 1, d.rect.width(), d.rect.height(), d.confidence
                else:
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                    conf = -1;

                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                _faces.append([x1, y1, x2, y2, w, h, conf])
                faces[i, :, :, :] = cv2.resize(frame[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

            results = model.predict(faces)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()
            # draw results
            for i, d in enumerate(detected):
                age = int(predicted_ages[i])
                gender = "M" if predicted_genders[i][0] < 0.5 else "F"
                label = "{}, {}".format(age, gender)
                _faces[i].append(age)
                _faces[i].append(gender)
                if hasattr(d, 'rect'):
                    draw_label(frame, (d.rect.left(), d.rect.top()), label)
                else:
                    draw_label(frame, (d.left(), d.top()), label)

        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000

        if (return_image == True):
            _, binframe = cv2.imencode('.jpg', frame)
            base64_bytes = b64encode(binframe)
            base64_string = base64_bytes.decode('utf-8')
            return jsonify({ "face_detector": face_detector, "confidence": confThreshold, "inference_time": duration, "base64": base64_string, "faces": _faces, "image_size": [width, height] }), 200, {'ContentType': 'application/json'}

        return jsonify({ "face_detector": face_detector, "confidence": confThreshold, "inference_time": duration, "faces": _faces, "image_size": [width, height] }), 200, {'ContentType': 'application/json'}

    except Exception as exc:
        # 'errors': exc
        print(exc)
        return json.dumps({'errors': "error" }),\
            200, {'ContentType': 'application/json'}


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route("/")
@app.route("/health")
@app.route("/healthz")
def test():
    try:
        return json.dumps({'success': True}), 200,\
                {'ContentType': 'application/json'}
    except Exception as exc:
        # 'errors': exc
        return json.dumps({'success': False }), 200,\
            {'ContentType': 'application/json'}

if __name__ == '__main__':
        port = config["port"]
        if os.environ.get('PORT') is not None:
            port = os.environ.get('PORT')
        print("running on port", port)
        app.run(host=config["host"], port=port, debug=False)
