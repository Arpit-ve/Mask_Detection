# detect_mask_video.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("mask_detector.model")

prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Labels
labels_dict = {0: "Mask", 1: "No Mask"}
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))  # ✅ Resize to speed up
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding boxes fall within the dimensions of the frame
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            # Extract face ROI
            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face = cv2.resize(face, (224, 224))
            face = face.astype("float") / 255.0
            face = np.expand_dims(face, axis=0)

            # Predict mask/no mask
            preds = model.predict(face, verbose=0)
            label = 0 if preds[0][0] < 0.5 else 1

            label_text = labels_dict[label]
            color = color_dict[label]

            # Display
            cv2.putText(frame, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Face Mask Detector", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

