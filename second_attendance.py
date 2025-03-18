import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from deepface import DeepFace
import datetime
import os

# Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.4)  # Lowered threshold

# Load stored face encodings
if os.path.exists("encodings.pickle"):
    import pickle
    with open("encodings.pickle", "rb") as f:
        known_encodings = pickle.load(f)
else:
    print("No face encodings found! Run face_encoding.py first.")
    exit()

# Load attendance file or create new one
attendance_file = "attendance.xlsx"
if os.path.exists(attendance_file):
    df = pd.read_excel(attendance_file)
else:
    df = pd.DataFrame(columns=["Name", "Date", "Time"])

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, box_w, box_h = (int(bboxC.xmin * w), int(bboxC.ymin * h),
                                  int(bboxC.width * w), int(bboxC.height * h))
            
            # Expand bounding box margin for better detection
            margin = 20
            x = max(0, x - margin)
            y = max(0, y - margin)
            box_w = min(w - x, box_w + 2 * margin)
            box_h = min(h - y, box_h + 2 * margin)
            
            # Extract and save face
            face_img = frame[y:y + box_h, x:x + box_w]
            temp_face_path = "temp_face.jpg"
            cv2.imwrite(temp_face_path, face_img)
            
            try:
                face_encoding = DeepFace.represent(temp_face_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                
                # Compare with known faces
                min_distance = float("inf")
                best_match = None
                for name, known_encoding in known_encodings.items():
                    distance = np.linalg.norm(np.array(face_encoding) - np.array(known_encoding))
                    if distance < min_distance:
                        min_distance = distance
                        best_match = name
                
                # Threshold for recognition
                if min_distance < 0.6:  # Adjust threshold as needed
                    print(f"Recognized: {best_match}")
                    cv2.putText(frame, best_match, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Mark attendance instantly
                    now = datetime.datetime.now()
                    df = df.append({"Name": best_match, "Date": now.strftime("%Y-%m-%d"), "Time": now.strftime("%H:%M:%S")}, ignore_index=True)
                    df.to_excel(attendance_file, index=False)
                
            except Exception as e:
                print(f"Recognition error: {e}")
    
    # Show video feed
    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
