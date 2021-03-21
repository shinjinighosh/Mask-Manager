import os
import sys
import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


from IPython.display import Image

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


import imutils
import time

ROOT_DIR = os.getcwd()
PATH_FACEDETECTOR = "resnet_ssd_facedetector"
confidence_min = 0.25

def f(imgPath):

    # load face detector model from disk
    print("[INFO] Loading face detector model...")
    prototxtPath = os.path.sep.join(
        [ROOT_DIR, PATH_FACEDETECTOR, "deploy.prototxt"])
    weightsPath = os.path.sep.join(
        [ROOT_DIR, PATH_FACEDETECTOR, "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] Loading mask detector model...")
    modelPath = os.path.sep.join([ROOT_DIR, "maskdetector.model"])
    model = load_model(modelPath)

    input_path = os.path.sep.join([ROOT_DIR, imgPath])
    image = cv2.imread(input_path)
    original = image.copy()
    (h, w) = image.shape[:2]

    # Construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    print("[INFO] Detecting faces...")
    net.setInput(blob)
    detections = net.forward()
    is_mask_there = []

    for i in range(0, detections.shape[2]):
        # Extract the confidence associated with the detection
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_min:
            # Compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract the face ROI, convert it from BGR to RGB channel, resize it to 224x224,
            # and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # Make prediction
            (mask, withoutMask) = model.predict(face)[0]

            label = "Mask" if mask > withoutMask else "No Mask"
            is_mask_there.append(label)
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # Display the label and bounding box rectangle on the output frame
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 10)

    # Save image
    OUTPUT_PATH = os.path.sep.join([ROOT_DIR, "output.jpg"])
    cv2.imwrite(OUTPUT_PATH, image)
    return "No Mask" in is_mask_there


# print(f("input.jpg"))
