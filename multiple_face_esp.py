import cv2
import mediapipe as mp
import pickle
import pandas as pd
import requests
import numpy as np
from datetime import datetime
from deepface import DeepFace

ENCODINGS_FILE = "encodings.pickle"
ATTENDANCE_FILE = "attendance.xlsx"
ESP32_CAM_URL = "http://10.10.53.116/capture"

# Load known face encodings
with open(ENCODINGS_FILE, "rb") as f:
    known_face_encodings = pickle.load(f)

# Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

# Load attendance
try:
    df = pd.read_excel(ATTENDANCE_FILE)
    attendance = {row["Name"]: row["Time"] for _, row in df.iterrows()}
except FileNotFoundError:
    attendance = {}

print("Press 'q' to exit")

while True:
    try:
        # Fetch the image from ESP32-CAM
        response = requests.get(ESP32_CAM_URL, timeout=5)
        img_array = np.frombuffer(response.content, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            print("Failed to decode frame")
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                # Extract face ROI
                face = frame[y:y+h, x:x+w]
                if face.size == 0:
                    continue

                # Save temp face image for DeepFace
                temp_face_path = "temp_face.jpg"
                cv2.imwrite(temp_face_path, face)

                # Recognize face
                try:
                    face_encoding = DeepFace.represent(temp_face_path, model_name="Facenet")[0]["embedding"]
                    best_match = None
                    best_distance = float("inf")

                    # Compare face encoding with stored encodings
                    for name, encodings in known_face_encodings.items():
                        for encoding in encodings:  # Iterate over multiple stored encodings
                            distance = sum((a - b) ** 2 for a, b in zip(face_encoding, encoding)) ** 0.5
                            if distance < best_distance:
                                best_distance = distance
                                best_match = name

                    if best_match and best_distance < 10:  # Adjust threshold if needed
                        name = best_match
                    else:
                        name = "Unknown"

                except Exception as e:
                    print(f"Recognition error: {e}")
                    name = "Unknown"

                # Mark attendance
                if name != "Unknown" and name not in attendance:
                    attendance[name] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    df = pd.DataFrame(list(attendance.items()), columns=["Name", "Time"])
                    df.to_excel(ATTENDANCE_FILE, index=False)
                    print(f"Attendance marked for {name}")

                # Display name on frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Show video feed
        cv2.imshow("Attendance System", frame)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching frame: {e}")

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
