import os

import cv2
import joblib
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
            face_images = frame[y1 : y1 + h, x1 : x1 + w, :]
            resized_face_images = cv2.resize(face_images, (50, 50))

            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

            if (i - 10) and len(face_data) < 10:
                face_data.append(resized_face_images)
            else:
                break

        cv2.imshow("frames", frame)

        if cv2.waitKey(1) == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()


face_data = np.asarray(face_data)
face_data = face_data.reshape(10, -1)
if "names.pkl" not in os.listdir(data_dir):
    names = [name] * 10
    with open(data_dir + "/names.pkl", "wb") as file:
        joblib.dump(names, file)

else:
    with open(data_dir + "/names.pkl", "rb") as file:
        names = joblib.load(file)

    with open(data_dir + "/names.pkl", "wb") as file:
        names = names + [name] * 10
        joblib.dump(names, file)

if "faces.pkl" not in os.listdir(data_dir):
    with open(data_dir + "/faces.pkl", "wb") as file:
        joblib.dump(face_data, file)

else:
    with open(data_dir + "/faces.pkl", "rb") as file:
        faces = joblib.load(file)
    with open(data_dir + "/faces.pkl", "wb") as file:
        faces = np.append(faces, face_data, 0)
        joblib.dump(faces, file)
