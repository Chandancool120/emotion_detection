
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array


CLASSES = ['angry','disgust','fear','happy','sad','surprise']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# load our serialized model from disk
print("[INFO] loading model...")
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# initialize the video stream, allow the camera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	face_detection = cv2.CascadeClassifier('detect_face.xml')
	faces = face_detection.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)	
	#blob = cv2.dnn.blobFromImage(cv2.resize(frame, (224, 224)),
	#	0.007843, (224, 224), 127.5)
	if len(faces)>0:
		for face in faces:
			face_image = frame[face[0]:face[0]+face[2],face[1]:face[1]+face[3]]
			if face_image.any():
				image = cv2.resize(face_image,(224,224))
				image = [img_to_array(image)]
				blob = np.array(image, dtype="float") / 255.0     
				# pass the blob through the network and obtain the detections and
				# predictions
				detections = loaded_model.predict(blob)		
				i,j = np.unravel_index(detections.argmax(), detections.shape)      

				# draw the prediction on the frame
				label = CLASSES[j]
				startX = face[0]
				startY = face[1]
				endX = face[0]+face[2]
				endY = face[1]+face[3]
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					COLORS[j], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[j], 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
