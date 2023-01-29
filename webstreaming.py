# import the necessary packages
from singlemotiondetector import SingleMotionDetector
#import pyimagesearch.motion_detection
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
from datetime import datetime, timedelta
import imutils
import time
import cv2
import numpy as np

#for saving to disk
import os
import random
import sys
from datetime import date

#make a new directory
name = random.randint(0, 1000)
print(name)
if os.path.isdir(str(name)) is False:
    name = random.randint(0, 1000)
    name = str(name)

today = date.today()
name = os.path.join(os.getcwd(), str(today) + '_' + str(name))
print("Logs saved in dir:", name)
os.mkdir(name)
cur_dir = os.path.dirname(os.path.abspath(sys.argv[0]))


start = datetime.now()
video_file_count = 1
video_file = os.path.join(name, str(video_file_count).zfill(4) + '_' + start.strftime("%H-%M-%S") + ".avi")

start_recording = False

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)

# initialize the FourCC, video writer, dimensions of the frame, and
# zeros array
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = None
(h, w) = (None, None)
zeros = None

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def detect_motion(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock, writer, zeros, video_file, name, start, video_file_count, start_recording
	# initialize the motion detector and the total number of frames
	# read thus far
	md = SingleMotionDetector(accumWeight=0.1)
	total = 0
	writer = None

    # loop over frames from the video stream
	while True:
		start_time = datetime.now()
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		frame = vs.read()
		frame = imutils.resize(frame, width=400)
		if zeros is None:
			(h, w) = frame.shape[:2]
			zeros = np.zeros((h, w), dtype="uint8")

		if (writer is None) & start_recording:
			# store the image dimensions, initialize the video writer,
			# and construct the zeros array
			start = datetime.now()
			writer = cv2.VideoWriter(video_file, fourcc, 15.0, (w * 2, h * 2), True)
        
		if (datetime.now() > (timedelta(minutes = 1) + start) ) & start_recording:
			start = datetime.now()
			video_file_count += 1
			video_file = os.path.join(name, str(video_file_count).zfill(4) + '_' + start.strftime("%H:%M:%S") + ".avi")
			writer = cv2.VideoWriter(video_file, fourcc, 15.0, (w * 2, h * 2), True)
			# No sleeping! We don't want to sleep, we want to write
			# time.sleep(10)    

		 # break the image into its RGB components, then construct the
		# RGB representation of each frame individually
		(B, G, R) = cv2.split(frame)
		R = cv2.merge([zeros, zeros, R])
		G = cv2.merge([zeros, G, zeros])
		B = cv2.merge([B, zeros, zeros])
		# construct the final output frame, storing the original frame
		# at the top-left, the red channel in the top-right, the green
		# channel in the bottom-right, and the blue channel in the
		# bottom-left
		output = np.zeros((h * 2, w * 2, 3), dtype="uint8")
		output[0:h, 0:w] = frame
		output[0:h, w:w * 2] = R
		output[h:h * 2, w:w * 2] = G
		output[h:h * 2, 0:w] = B
		# write the output frame to file
		if start_recording:
			writer.write(output)    

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (7, 7), 0)
		# grab the current timestamp and draw it on the frame
		timestamp = datetime.now()
		cv2.putText(frame, timestamp.strftime(
			"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
        		# if the total number of frames has reached a sufficient
		# number to construct a reasonable background model, then
		# continue to process the frame
		if total > frameCount:
			# detect motion in the image
			motion = md.detect(gray)
			# check to see if motion was found in the frame
			if motion is not None:
				# unpack the tuple and draw the box surrounding the
				# "motion area" on the output frame
				(thresh, (minX, minY, maxX, maxY)) = motion
				cv2.rectangle(frame, (minX, minY), (maxX, maxY),
					(0, 0, 255), 2)
		
		# update the background model and increment the total number
		# of frames read thus far
		md.update(gray)
		total += 1
		# acquire the lock, set the output frame, and release the
		# lock
		with lock:
			outputFrame = frame.copy()

        
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock
	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue
			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			# ensure the frame was successfully encoded
			if not flag:
				continue
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

#background process happening without any refreshing
@app.route('/background_process_test')
def background_process_test():
    global start_recording
    print("hello")
    start_recording = True
    return ("nothing")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())
	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()
	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)
# release the video stream pointer
vs.stop()
writer.release()          