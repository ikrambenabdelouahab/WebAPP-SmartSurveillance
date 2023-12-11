from imutils.video import VideoStream
from flask import Response
from flask import Flask, jsonify
from flask import render_template
import threading
import imutils
import time
import cv2
import numpy as np
from scipy.spatial import distance
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import os
from tensorflow import Graph, Session
from tensorflow.keras import backend as K

outputFrame = None
outputFrame_2 = None
outputFrame_3 = None
lock = threading.Lock()
app = Flask(__name__, static_folder='static', template_folder='templates')

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# Object Detection model
net = cv2.dnn.readNetFromCaffe('src/MobileNetSSD_deploy.prototxt.txt', 'src/MobileNetSSD_deploy.caffemodel')
print("[INFO] SSD model Loaded...")

# Face detector model
faceNet = cv2.dnn.readNetFromCaffe('src/deploy.prototxt.txt', 'src/res10_300x300_ssd_iter_140000.caffemodel')
print("[INFO] Face detection model Loaded...")
# Facemask classifier 
global maskNet
graph1 = Graph()
with graph1.as_default():
	session1 = Session(graph=graph1)
	with session1.as_default():
		maskNet = load_model("src/mask_detector.model")

vs = VideoStream(src=0).start()

time.sleep(2.0)
frame_id = 0
cnt = 0
detection_id = 1

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/social_distance")
def social_distance():
	return render_template("social_distance.html")

@app.route("/facemask_detector")
def facemask_detector():
	return render_template("facemask_detector.html")

@app.route("/live_stream")
def live_stream():
	return render_template("live_stream.html")

@app.route("/lecteur_qr")
def lecteur_qr():
	return render_template("lecteur_qr.html")


def detect_motion(frameCount):
	global vs, outputFrame, lock, frame_id, cnt, detection_id, num
	while True:
		frame = vs.read()
		frame = imutils.resize(frame, width=400)
		frame_id += 1
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
		net.setInput(blob)
		detections = net.forward()
		midpoints = []
		label = ""
		color = (0,255,0)
		for i in np.arange(0, detections.shape[2]):
		    confidence = detections[0, 0, i, 2]
		    if confidence > 0.2:
		        idx = int(detections[0, 0, i, 1])
		        if idx == 15:
		            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		            (startX, startY, endX, endY) = box.astype("int")
		            cnt += 1
		            midp = (endX, endY)
		            midpoints.append([midp, cnt])
		            num = len(midpoints)
		            #print(num)

		            for m in range(num):
		                for n in range(m + 1, num):
		                    if m != n:
		                        dst = distance.euclidean(midpoints[m][0], midpoints[n][0])
		                        p1 = midpoints[m][1]
		                        p2 = midpoints[n][1]
		                        #print("Distance entre ", p1, " et ", p2, " ==== ", int(dst))
		                        if (dst <=200):
		                        	#print("ALERT")
		                        	label = "ALERT"
		                        	color = (0,0,255)
		                        else:
		                        	#print("Normal")
		                        	label = "Good"
		                        	color = (0,255,0)

		            #label = "{}: {}%".format(CLASSES[idx],confidence * 100)
		            # boxes colors BRG : (255,0,0), (0,255,0), (0,0,255)
		            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		            y = startY - 15 if startY - 15 > 15 else startY + 15
		            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		# acquire the lock, set the output frame, and release the lock
		with lock:
		    outputFrame = frame.copy()

def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")


def live_stream(frameCount):
	global vs, outputFrame_2, lock
	while True:
		frame = vs.read()
		frame = imutils.resize(frame, width=400)
		outputframe_2 = frame
		with lock:
		    outputFrame_2 = frame.copy()

def generate_live():
    global outputFrame_2, lock
    while True:
        with lock:
            if outputFrame_2 is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame_2)
            if not flag:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

@app.route("/live")
def live():
	return Response(generate_live(), mimetype = "multipart/x-mixed-replace; boundary=frame")


def facemask_detector(frameCount):
	global vs, outputFrame_3, lock
	while True:
		frame = vs.read()
		frame = imutils.resize(frame, width=400)
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))

		faceNet.setInput(blob)
		detections = faceNet.forward()
		# print(detections.shape)
		faces = []
		locs = []
		preds = []

		for i in range(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > 0.5:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				(startX, startY) = (max(0, startX), max(0, startY))
				(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

				face = frame[startY:endY, startX:endX]
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				faces.append(face)
				locs.append((startX, startY, endX, endY))

				faces = np.array(faces, dtype="float32")
				K.set_session(session1)
				with graph1.as_default():
					preds = maskNet.predict(faces, batch_size=32)
	
		# only make a predictions if at least one face was detected
		#if len(faces) > 0:
		#	faces = np.array(faces, dtype="float32")
		#	preds = maskNet.predict(faces, batch_size=32)
			
		# loop over the detected face locations and their corresponding locations
		for (box, pred) in zip(locs, preds):
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred

			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

			cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		with lock:
		    outputFrame_3 = frame.copy()

def generate_facemask():
    global outputFrame_3, lock
    while True:
        with lock:
            if outputFrame_3 is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame_3)
            if not flag:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

@app.route("/facemask")
def facemask():
	return Response(generate_facemask(), mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
	t = threading.Thread(target=detect_motion, args=(32,))
	t.daemon = True
	t.start()

	t_2 = threading.Thread(target=live_stream, args=(32,))
	t_2.daemon = True
	t_2.start()

	t_3 = threading.Thread(target=facemask_detector, args=(32,))
	t_3.daemon = True
	t_3.start()

	app.run(host="127.0.0.1", port="8000", debug=True, threaded=True, use_reloader=False)
vs.stop()
