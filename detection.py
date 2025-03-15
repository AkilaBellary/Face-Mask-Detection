# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import sys
import io

# Fix encoding issue
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def detect_and_predict_mask(frame, faceCascade, maskNet):
    # convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces_list = []
    locs = []
    preds = []

    # loop over the detected faces
    for (x, y, w, h) in faces:
        # extract the face ROI, convert it from BGR to RGB channel
        # ordering, resize it to 224x224, and preprocess it
        face = frame[y:y + h, x:x + w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)

        # add the face and bounding boxes to their respective lists
        faces_list.append(face)
        locs.append((x, y, x + w, y + h))

    # only make predictions if at least one face was detected
    if len(faces_list) > 0:
        # for faster inference, make batch predictions on all faces
        faces_list = np.array(faces_list, dtype="float32")
        preds = maskNet.predict(faces_list, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding predictions
    return (locs, preds)

# load the Haar Cascade face detector
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# load the face mask detector model from disk
maskNet = load_model("mask_detector.h5")

# Recompile the model to suppress warnings
maskNet.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # detect faces in the frame and determine if they are wearing a mask
    (locs, preds) = detect_and_predict_mask(frame, faceCascade, maskNet)

    # loop over the detected face locations and their corresponding predictions
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()