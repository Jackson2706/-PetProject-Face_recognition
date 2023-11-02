import os
import pickle

import cv2
import numpy as np

data_dir = "./dataset"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

cap = cv2.VideoCapture(0)
facecascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face_data = []
i = 0

if not cap.isOpened():
    print(f"Unable to open camera")
else:
    name = input("Enter name: -->\t")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to read frame from camera")
            break

        # Convert to gray scale
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        face_coordinates = facecascade.detectMultiScale(gray_scale, 1.3, 4)

        for x1, y1, w, h in face_coordinates:
            face_images = frame[x1 : x1 + w, y1 : y1 + h]
            resized_face_images = cv2.resize(face_images, (50, 50))

            if (i - 10) and len(face_data) < 10:
                face_data.append(resized_face_images)
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

        cv2.imshow("frames", frame)

        if cv2.waitKey(1) == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()

try:
    face_data = np.asarray(face_data)
    face_data = face_data.reshape(10, -1)

    if "name.pkl" not in os.listdir(data_dir):
        names = [name] * 10
        with open(data_dir + "/name.pkl", "wb") as file:
            pickle.dump(names, file)

    else:
        with open(data_dir + "/name.pkl", "wb") as file:
            names = pickle.load(file)
            names = names + [name] * 10
            pickle.dump(names, file)

    if "faces.pkl" not in os.listdir(data_dir):
        with open(data_dir + "/faces.pkl", "wb") as file:
            pickle.dump(face_data, file)

    else:
        with open(data_dir + "/faces.pkl", "wb") as file:
            faces = pickle.load(file)
            faces = np.append(faces, face_data, 0)
            pickle.dump(faces, file)
except:
    print("Failed to create new data")
