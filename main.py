import cv2
import joblib
from sklearn.neighbors import KNeighborsClassifier

cap = cv2.VideoCapture(0)
facecascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load dataset
with open("dataset/names.pkl", "rb") as file:
    labels = joblib.load(file)

with open("dataset/faces.pkl", "rb") as file:
    faces = joblib.load(file)

# Defining model
knn = KNeighborsClassifier()
knn.fit(faces, labels)

if not cap.isOpened():
    print(f"Unable to open camera")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to read frame from camera")
            break

        # Convert to gray scale
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        face_coordinates = facecascade.detectMultiScale(gray_scale, 1.3, 4)

        for [x, y, w, h] in face_coordinates:
            face_input = frame[y : y + h, x : x + w, :]
            r = cv2.resize(face_input, (50, 50)).flatten().reshape(1, -1)
            text = knn.predict(r)
            cv2.putText(
                frame,
                text[0],
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        print(f"Face dectections: {face_coordinates}")
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    cap.release()
    cv2.destroyAllWindows()
